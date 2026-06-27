//! gRPC server for the reverse-mode agent protocol.
//!
//! The server drives the ReAct loop (calls LLM, decides actions).
//! When it needs a tool executed, it sends a ToolCallRequest to the
//! connected agent worker via gRPC bidirectional streaming.
//!
//! Usage:
//!   1. Implement `AgentHandler` to provide your ReAct logic
//!   2. Call `serve(addr, handler)` to start the gRPC server
//!   3. Agent workers connect via `AgentService::Session` RPC

use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use cyber_agent_proto::{
    AgentMessage, AgentService, AgentServiceServer, ServerMessage,
    ToolCallRequest, ToolCallResult, ToolManifest,
    agent_message::Payload,
};

/// Implement this trait to provide the server-side ReAct logic.
///
/// The handler receives the tool manifest from the agent, then
/// drives the conversation by calling `call_tool` on the provided
/// `ToolExecutor` as many times as needed.
#[async_trait::async_trait]
pub trait AgentHandler: Send + Sync + 'static {
    async fn handle_session(
        &self,
        manifest: ToolManifest,
        executor: Box<dyn ToolExecutor>,
    ) -> Result<String, Status>;
}

/// Interface for calling tools on the remote agent.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn call_tool(
        &self,
        id: &str,
        name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult, Status>;
}

struct GrpcToolExecutor {
    tx: mpsc::Sender<ServerMessage>,
    result_rx: tokio::sync::Mutex<mpsc::Receiver<ToolCallResult>>,
}

#[async_trait::async_trait]
impl ToolExecutor for GrpcToolExecutor {
    async fn call_tool(
        &self,
        id: &str,
        name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult, Status> {
        let req = ToolCallRequest {
            id: id.into(),
            name: name.into(),
            arguments_json: arguments_json.into(),
            done: false,
            final_text: String::new(),
        };
        self.tx
            .send(ServerMessage { request: Some(req) })
            .await
            .map_err(|_| Status::internal("agent disconnected"))?;

        self.result_rx
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| Status::internal("agent closed stream"))
    }
}

struct AgentServiceImpl {
    handler: Arc<dyn AgentHandler>,
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

        // Wait for the first message: must be a ToolManifest
        let manifest = match inbound.next().await {
            Some(Ok(msg)) => match msg.payload {
                Some(Payload::Manifest(m)) => m,
                _ => return Err(Status::invalid_argument("first message must be ToolManifest")),
            },
            Some(Err(e)) => return Err(e),
            None => return Err(Status::cancelled("stream closed before manifest")),
        };

        // Channels for server → agent (tool call requests)
        let (server_tx, server_rx) = mpsc::channel::<ServerMessage>(32);
        // Channel for agent → server (tool call results, forwarded from inbound stream)
        let (result_tx, result_rx) = mpsc::channel::<ToolCallResult>(32);

        // Forward inbound tool results to the result channel
        tokio::spawn(async move {
            while let Some(Ok(msg)) = inbound.next().await {
                if let Some(Payload::Result(result)) = msg.payload {
                    if result_tx.send(result).await.is_err() {
                        break;
                    }
                }
            }
        });

        let executor = Box::new(GrpcToolExecutor {
            tx: server_tx.clone(),
            result_rx: tokio::sync::Mutex::new(result_rx),
        });

        // Run the handler in a background task
        let handler = self.handler.clone();
        tokio::spawn(async move {
            let final_text = match handler.handle_session(manifest, executor).await {
                Ok(text) => text,
                Err(e) => format!("handler error: {}", e),
            };

            // Send done signal
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

/// Start the gRPC server.
pub async fn serve(
    addr: std::net::SocketAddr,
    handler: Arc<dyn AgentHandler>,
) -> Result<(), tonic::transport::Error> {
    let service = AgentServiceImpl { handler };
    tonic::transport::Server::builder()
        .add_service(AgentServiceServer::new(service))
        .serve(addr)
        .await
}
