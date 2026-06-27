//! gRPC transport adapter for the agent server.
//!
//! Wraps tonic bidi streaming as a `Transport`, then delegates
//! to the transport-agnostic `serve_session()`.

use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use cyber_agent_proto::{
    AgentMessage, AgentService, AgentServiceServer, ServerMessage,
    ToolCallRequest,
    agent_message::Payload,
};

use crate::AgentHandler;

/// Start a gRPC server that accepts agent sessions.
pub async fn serve_grpc(
    addr: std::net::SocketAddr,
    handler: Arc<dyn AgentHandler + 'static>,
) -> Result<(), tonic::transport::Error> {
    let service = AgentServiceImpl { handler };
    tonic::transport::Server::builder()
        .add_service(AgentServiceServer::new(service))
        .serve(addr)
        .await
}

struct AgentServiceImpl {
    handler: Arc<dyn AgentHandler + 'static>,
}

#[tonic::async_trait]
impl AgentService for AgentServiceImpl {
    type SessionStream =
        Pin<Box<dyn futures_core::Stream<Item = Result<ServerMessage, Status>> + Send>>;

    async fn session(
        &self,
        request: Request<Streaming<AgentMessage>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();

        let manifest = match inbound.next().await {
            Some(Ok(msg)) => match msg.payload {
                Some(Payload::Manifest(m)) => m,
                _ => return Err(Status::invalid_argument("first message must be ToolManifest")),
            },
            Some(Err(e)) => return Err(e),
            None => return Err(Status::cancelled("stream closed before manifest")),
        };

        let (server_tx, server_rx) = mpsc::channel::<ServerMessage>(32);
        let (result_tx, result_rx) = mpsc::channel::<Vec<u8>>(32);

        tokio::spawn(async move {
            while let Some(Ok(msg)) = inbound.next().await {
                if let Some(Payload::Result(result)) = msg.payload {
                    let bytes = serde_json::to_vec(&result).unwrap_or_default();
                    if result_tx.send(bytes).await.is_err() {
                        break;
                    }
                }
            }
        });

        let transport = Arc::new(GrpcSessionTransport {
            tx: server_tx.clone(),
            rx: tokio::sync::Mutex::new(result_rx),
        });

        let handler = self.handler.clone();
        tokio::spawn(async move {
            let executor = crate::TransportExecutor {
                transport: transport.as_ref(),
            };

            let final_text = match handler.handle_session(manifest, &executor).await {
                Ok(text) => text,
                Err(e) => format!("handler error: {}", e),
            };

            let _ = server_tx
                .send(ServerMessage {
                    request: Some(ToolCallRequest {
                        done: true,
                        final_text,
                        ..Default::default()
                    }),
                })
                .await;
        });

        let output_stream = ReceiverStream::new(server_rx).map(Ok);
        Ok(Response::new(Box::pin(output_stream)))
    }
}

struct GrpcSessionTransport {
    tx: mpsc::Sender<ServerMessage>,
    rx: tokio::sync::Mutex<mpsc::Receiver<Vec<u8>>>,
}

#[async_trait::async_trait]
impl cyber_agent_proto::Transport for GrpcSessionTransport {
    async fn send(&self, data: &[u8]) -> anyhow::Result<()> {
        let req: ToolCallRequest = serde_json::from_slice(data)?;
        self.tx
            .send(ServerMessage { request: Some(req) })
            .await
            .map_err(|_| anyhow::anyhow!("agent disconnected"))
    }

    async fn recv(&self) -> anyhow::Result<Vec<u8>> {
        self.rx
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("agent closed stream"))
    }
}
