//! gRPC client for the worker — connects to server, executes tool calls.

use anyhow::Result;
use tokio_stream::StreamExt;

use cyber_agent_proto::{
    AgentMessage, AgentServiceClient, ToolCallResult, ToolManifest,
    agent_message::Payload,
};
use cyber_agent_tool::ToolRegistry;

use crate::WorkerResult;

pub async fn run_grpc_worker(
    server_url: &str,
    tools: &ToolRegistry,
    session: &str,
) -> Result<WorkerResult> {
    let mut client = AgentServiceClient::connect(server_url.to_string()).await?;

    let (tx, rx) = tokio::sync::mpsc::channel::<AgentMessage>(32);

    // Send manifest as the first message
    let manifest = ToolManifest {
        session: session.into(),
        tools: tools.list_tool_defs(),
    };
    tx.send(AgentMessage {
        payload: Some(Payload::Manifest(manifest)),
    })
    .await?;

    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let response = client.session(outbound).await?;
    let mut inbound = response.into_inner();

    let mut tool_calls_executed = 0usize;

    while let Some(msg) = inbound.next().await {
        let msg = msg?;
        let req = match msg.request {
            Some(r) => r,
            None => continue,
        };

        if req.done {
            return Ok(WorkerResult {
                final_text: req.final_text,
                tool_calls_executed,
            });
        }

        // Execute the tool
        let result = match tools.get(&req.name) {
            Some(tool) => {
                let args: serde_json::Value =
                    serde_json::from_str(&req.arguments_json).unwrap_or(serde_json::json!({}));
                match tool.execute(args).await {
                    Ok(val) => ToolCallResult {
                        id: req.id.clone(),
                        success: true,
                        result_json: serde_json::to_string(&val).unwrap_or_default(),
                        error: String::new(),
                    },
                    Err(e) => ToolCallResult {
                        id: req.id.clone(),
                        success: false,
                        result_json: String::new(),
                        error: e.to_string(),
                    },
                }
            }
            None => ToolCallResult {
                id: req.id.clone(),
                success: false,
                result_json: String::new(),
                error: format!("unknown tool: {}", req.name),
            },
        };

        tool_calls_executed += 1;

        tx.send(AgentMessage {
            payload: Some(Payload::Result(result)),
        })
        .await
        .map_err(|e| anyhow::anyhow!("send result: {}", e))?;
    }

    Ok(WorkerResult {
        final_text: String::new(),
        tool_calls_executed,
    })
}
