//! Integration test for the reverse-mode worker protocol.
//!
//! Simulates a server that drives the ReAct loop:
//!   1. Receives tool manifest from agent
//!   2. Sends tool call requests
//!   3. Receives tool results
//!   4. Sends done signal

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use cyber_agent_proto::{ToolCallRequest, ToolCallResult, ToolManifest};
use cyber_agent_tool::{AgentTool, ToolRegistry};
use cyber_agent_worker::{WorkerTransport, run_worker_loop_bidi};

// ── Channel-based transport ─────────────────────────────────────────────

struct ChannelTransport {
    tx: mpsc::Sender<Vec<u8>>,
    rx: tokio::sync::Mutex<mpsc::Receiver<Vec<u8>>>,
}

#[async_trait]
impl WorkerTransport for ChannelTransport {
    async fn send(&self, data: &[u8]) -> Result<()> {
        self.tx.send(data.to_vec()).await.map_err(|e| anyhow::anyhow!("{}", e))
    }
    async fn recv(&self) -> Result<Vec<u8>> {
        self.rx.lock().await.recv().await.ok_or_else(|| anyhow::anyhow!("channel closed"))
    }
}

fn channel_pair() -> (Arc<ChannelTransport>, Arc<ChannelTransport>) {
    let (a_tx, b_rx) = mpsc::channel::<Vec<u8>>(32);
    let (b_tx, a_rx) = mpsc::channel::<Vec<u8>>(32);
    let agent_side = Arc::new(ChannelTransport {
        tx: a_tx,
        rx: tokio::sync::Mutex::new(a_rx),
    });
    let server_side = Arc::new(ChannelTransport {
        tx: b_tx,
        rx: tokio::sync::Mutex::new(b_rx),
    });
    (agent_side, server_side)
}

// ── Test tools ──────────────────────────────────────────────────────────

struct ShellTool;

#[async_trait]
impl AgentTool for ShellTool {
    fn name(&self) -> &str { "shell" }
    fn description(&self) -> &str { "Execute a shell command" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let cmd = params.get("command").and_then(|v| v.as_str()).unwrap_or("echo noop");
        let out = std::process::Command::new("sh").args(["-c", cmd]).output()?;
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        Ok(serde_json::json!({"stdout": stdout, "exit_code": out.status.code().unwrap_or(-1)}))
    }
}

struct CalcTool;

#[async_trait]
impl AgentTool for CalcTool {
    fn name(&self) -> &str { "calc" }
    fn description(&self) -> &str { "Add two numbers" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let a = params["a"].as_i64().unwrap_or(0);
        let b = params["b"].as_i64().unwrap_or(0);
        Ok(serde_json::json!({"result": a + b}))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

/// Full reverse-mode test: server drives the loop, agent executes tools.
///
/// Server-side logic (simulated):
///   1. Receive manifest, verify tools
///   2. Call "shell" to get hostname
///   3. Call "calc" to add 17 + 25
///   4. Send done with combined answer
#[tokio::test]
async fn reverse_mode_multi_tool() {
    let (agent_transport, server_transport) = channel_pair();

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));
    tools.register(Box::new(CalcTool));

    // Spawn the agent worker
    let agent_handle = tokio::spawn(async move {
        run_worker_loop_bidi(agent_transport.as_ref(), &tools, "test-session").await
    });

    // Server side: simulate ReAct loop
    let server_handle = tokio::spawn(async move {
        let server = server_transport;

        // 1. Receive manifest
        let manifest_bytes = server.recv().await.unwrap();
        let manifest: ToolManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        eprintln!("[server] received manifest: session={}, tools={:?}",
            manifest.session,
            manifest.tools.iter().map(|t| &t.name).collect::<Vec<_>>()
        );
        assert_eq!(manifest.session, "test-session");
        assert_eq!(manifest.tools.len(), 2);

        // 2. Call shell tool
        let shell_req = ToolCallRequest {
            id: "call_001".into(),
            name: "shell".into(),
            arguments_json: r#"{"command":"echo hello_reverse"}"#.into(),
            done: false,
            final_text: String::new(),
        };
        server.send(&serde_json::to_vec(&shell_req).unwrap()).await.unwrap();
        eprintln!("[server] sent shell call");

        // Receive shell result
        let result_bytes = server.recv().await.unwrap();
        let shell_result: ToolCallResult = serde_json::from_slice(&result_bytes).unwrap();
        eprintln!("[server] shell result: success={}, result={}", shell_result.success, shell_result.result_json);
        assert!(shell_result.success);
        assert_eq!(shell_result.id, "call_001");
        let shell_output: serde_json::Value = serde_json::from_str(&shell_result.result_json).unwrap();
        assert_eq!(shell_output["stdout"], "hello_reverse");

        // 3. Call calc tool
        let calc_req = ToolCallRequest {
            id: "call_002".into(),
            name: "calc".into(),
            arguments_json: r#"{"a":17,"b":25}"#.into(),
            done: false,
            final_text: String::new(),
        };
        server.send(&serde_json::to_vec(&calc_req).unwrap()).await.unwrap();
        eprintln!("[server] sent calc call");

        let result_bytes = server.recv().await.unwrap();
        let calc_result: ToolCallResult = serde_json::from_slice(&result_bytes).unwrap();
        eprintln!("[server] calc result: success={}, result={}", calc_result.success, calc_result.result_json);
        assert!(calc_result.success);
        let calc_output: serde_json::Value = serde_json::from_str(&calc_result.result_json).unwrap();
        assert_eq!(calc_output["result"], 42);

        // 4. Send done
        let done_req = ToolCallRequest {
            id: String::new(),
            name: String::new(),
            arguments_json: String::new(),
            done: true,
            final_text: "The hostname echoed hello_reverse and 17+25=42.".into(),
        };
        server.send(&serde_json::to_vec(&done_req).unwrap()).await.unwrap();
        eprintln!("[server] sent done");
    });

    // Wait for both sides
    server_handle.await.unwrap();
    let result = agent_handle.await.unwrap().unwrap();

    eprintln!("[result] final_text: {}", result.final_text);
    eprintln!("[result] tool_calls_executed: {}", result.tool_calls_executed);

    assert_eq!(result.tool_calls_executed, 2);
    assert!(result.final_text.contains("42"));
    eprintln!("\n=== REVERSE MODE TEST PASSED ===\n");
}

/// Test: server calls a nonexistent tool, agent returns error gracefully.
#[tokio::test]
async fn reverse_mode_unknown_tool() {
    let (agent_transport, server_transport) = channel_pair();

    let tools = ToolRegistry::new(); // empty — no tools

    let agent_handle = tokio::spawn(async move {
        run_worker_loop_bidi(agent_transport.as_ref(), &tools, "s1").await
    });

    let server_handle = tokio::spawn(async move {
        let server = server_transport;

        // Receive manifest (empty)
        let manifest_bytes = server.recv().await.unwrap();
        let manifest: ToolManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        assert!(manifest.tools.is_empty());

        // Call nonexistent tool
        let req = ToolCallRequest {
            id: "bad_call".into(),
            name: "nonexistent".into(),
            arguments_json: "{}".into(),
            done: false,
            final_text: String::new(),
        };
        server.send(&serde_json::to_vec(&req).unwrap()).await.unwrap();

        // Should get error result
        let result_bytes = server.recv().await.unwrap();
        let result: ToolCallResult = serde_json::from_slice(&result_bytes).unwrap();
        assert!(!result.success);
        assert!(result.error.contains("unknown tool"));
        eprintln!("[server] got expected error: {}", result.error);

        // Done
        let done = ToolCallRequest { done: true, final_text: "aborted".into(), ..Default::default() };
        server.send(&serde_json::to_vec(&done).unwrap()).await.unwrap();
    });

    server_handle.await.unwrap();
    let result = agent_handle.await.unwrap().unwrap();
    assert_eq!(result.final_text, "aborted");
    assert_eq!(result.tool_calls_executed, 1);
    eprintln!("=== UNKNOWN TOOL TEST PASSED ===");
}
