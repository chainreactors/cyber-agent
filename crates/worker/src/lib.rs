//! Reverse-mode agent worker: server drives the ReAct loop,
//! the local agent only executes tool calls.
//!
//! Protocol flow:
//!   1. Agent sends ToolManifest (available tools) to server
//!   2. Server sends ToolCallRequest (execute this tool)
//!   3. Agent executes the tool, sends ToolCallResult back
//!   4. Repeat 2-3 until server sends done=true

use anyhow::{anyhow, Result};

use cyber_agent_proto::{ToolCallRequest, ToolManifest, Transport};
use cyber_agent_tool::ToolRegistry;

pub struct WorkerResult {
    pub final_text: String,
    pub tool_calls_executed: usize,
}

pub async fn run_worker(
    transport: &dyn Transport,
    tools: &ToolRegistry,
    session: &str,
) -> Result<WorkerResult> {
    let manifest = ToolManifest {
        session: session.into(),
        tools: tools.list_tool_defs(),
    };
    transport.send(&serde_json::to_vec(&manifest)?).await?;

    let mut tool_calls_executed = 0usize;

    loop {
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

        transport.send(&serde_json::to_vec(&result)?).await?;
    }
}
