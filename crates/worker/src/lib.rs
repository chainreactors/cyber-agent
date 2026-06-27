//! Reverse-mode agent worker: server drives the ReAct loop,
//! the local agent only executes tool calls.
//!
//! Protocol flow:
//!   1. Agent sends ToolManifest (available tools) to server
//!   2. Server sends ToolCallRequest (execute this tool)
//!   3. Agent executes the tool, sends ToolCallResult back
//!   4. Repeat 2-3 until server sends done=true
//!
//! The server can be written in any language — it controls the LLM,
//! reasoning, and action selection. The agent is a pure tool executor.

#[cfg(feature = "grpc")]
pub mod grpc;

use anyhow::{anyhow, Result};

use cyber_agent_proto::{ToolCallRequest, ToolManifest};
use cyber_agent_tool::ToolRegistry;
use cyber_agent_proto::Transport;

pub struct WorkerResult {
    pub final_text: String,
    pub tool_calls_executed: usize,
}

pub async fn run_worker_loop(
    transport: &dyn Transport,
    tools: &ToolRegistry,
    session: &str,
) -> Result<WorkerResult> {
    // Step 1: send tool manifest
    let manifest = ToolManifest {
        session: session.into(),
        tools: tools.list_tool_defs(),
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    transport.request(&manifest_bytes).await?;

    let mut tool_calls_executed = 0usize;

    // Step 2-4: tool call loop
    loop {
        // Receive tool call request from server
        let req_bytes = transport.request(&[]).await?;
        let req: ToolCallRequest = serde_json::from_slice(&req_bytes)?;

        // Check if session is complete
        if req.done {
            return Ok(WorkerResult {
                final_text: req.final_text,
                tool_calls_executed,
            });
        }

        let result = tools.execute_call(&req.id, &req.name, &req.arguments_json).await;

        tool_calls_executed += 1;

        // Send result back to server
        let result_bytes = serde_json::to_vec(&result)?;
        transport.request(&result_bytes).await?;
    }
}

/// Bidirectional transport for the worker protocol.
///
/// Unlike the provider's Transport which is request-response,
/// the worker needs to both send and receive independently.
/// This trait wraps a channel-based transport where `send` pushes
/// data and `recv` blocks until data arrives.
#[async_trait::async_trait]
pub trait WorkerTransport: Send + Sync {
    async fn send(&self, data: &[u8]) -> Result<()>;
    async fn recv(&self) -> Result<Vec<u8>>;
}

pub async fn run_worker_loop_bidi(
    transport: &dyn WorkerTransport,
    tools: &ToolRegistry,
    session: &str,
) -> Result<WorkerResult> {
    // Step 1: send tool manifest
    let manifest = ToolManifest {
        session: session.into(),
        tools: tools.list_tool_defs(),
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    transport.send(&manifest_bytes).await?;

    let mut tool_calls_executed = 0usize;

    // Step 2-4: tool call loop
    loop {
        // Receive tool call request from server
        let req_bytes = transport.recv().await?;
        let req: ToolCallRequest =
            serde_json::from_slice(&req_bytes).map_err(|e| anyhow!("parse ToolCallRequest: {}", e))?;

        if req.done {
            return Ok(WorkerResult {
                final_text: req.final_text,
                tool_calls_executed,
            });
        }

        let result = tools.execute_call(&req.id, &req.name, &req.arguments_json).await;

        tool_calls_executed += 1;

        let result_bytes = serde_json::to_vec(&result)?;
        transport.send(&result_bytes).await?;
    }
}
