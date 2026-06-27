//! Server-side agent session driver.
//!
//! Transport-agnostic: works over any `Transport` implementation
//! (TCP, WebSocket, gRPC, channels, C2, etc.)
//!
//! Usage:
//!   1. Implement `AgentHandler` to provide your ReAct logic
//!   2. Accept a connection, wrap it as `dyn Transport`
//!   3. Call `serve_session(transport, handler)` to drive the session

#[cfg(feature = "grpc")]
pub mod grpc;

use anyhow::{anyhow, Result};

use cyber_agent_proto::{
    ToolCallRequest, ToolCallResult, ToolManifest, Transport,
};

/// Implement this trait to provide the server-side ReAct logic.
#[async_trait::async_trait]
pub trait AgentHandler: Send + Sync {
    async fn handle_session(
        &self,
        manifest: ToolManifest,
        executor: &dyn ToolExecutor,
    ) -> Result<String>;
}

/// Interface for calling tools on the remote agent.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn call_tool(
        &self,
        id: &str,
        name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult>;
}

/// Transport-backed tool executor — sends ToolCallRequest, waits for ToolCallResult.
pub(crate) struct TransportExecutor<'a> {
    pub(crate) transport: &'a dyn Transport,
}

#[async_trait::async_trait]
impl ToolExecutor for TransportExecutor<'_> {
    async fn call_tool(
        &self,
        id: &str,
        name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        let req = ToolCallRequest {
            id: id.into(),
            name: name.into(),
            arguments_json: arguments_json.into(),
            done: false,
            final_text: String::new(),
        };
        self.transport
            .send(&serde_json::to_vec(&req)?)
            .await?;

        let result_bytes = self.transport.recv().await?;
        serde_json::from_slice(&result_bytes)
            .map_err(|e| anyhow!("parse ToolCallResult: {}", e))
    }
}

/// Drive a single agent session over any Transport.
///
/// 1. Receives ToolManifest from the agent
/// 2. Calls handler.handle_session() — handler uses executor to call tools
/// 3. Sends done signal to the agent
pub async fn serve_session(
    transport: &dyn Transport,
    handler: &dyn AgentHandler,
) -> Result<String> {
    let manifest_bytes = transport.recv().await?;
    let manifest: ToolManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| anyhow!("parse ToolManifest: {}", e))?;

    let executor = TransportExecutor { transport };

    let final_text = match handler.handle_session(manifest, &executor).await {
        Ok(text) => text,
        Err(e) => format!("handler error: {}", e),
    };

    let done = ToolCallRequest {
        done: true,
        final_text: final_text.clone(),
        ..Default::default()
    };
    transport.send(&serde_json::to_vec(&done)?).await?;

    Ok(final_text)
}
