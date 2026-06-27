//! Test serve_session() over pure channel transport — no gRPC involved.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use cyber_agent_proto::{ToolCallResult, ToolManifest, Transport};
use cyber_agent_server::{serve_session, AgentHandler, ToolExecutor};
use cyber_agent_tool::{AgentTool, ToolRegistry};
use cyber_agent_worker::run_worker;

// ── Channel Transport ───────────────────────────────────────────────────

struct ChannelTransport {
    tx: mpsc::Sender<Vec<u8>>,
    rx: tokio::sync::Mutex<mpsc::Receiver<Vec<u8>>>,
}

#[async_trait]
impl Transport for ChannelTransport {
    async fn send(&self, data: &[u8]) -> Result<()> {
        self.tx.send(data.to_vec()).await.map_err(|e| anyhow::anyhow!("{}", e))
    }
    async fn recv(&self) -> Result<Vec<u8>> {
        self.rx.lock().await.recv().await.ok_or_else(|| anyhow::anyhow!("closed"))
    }
}

fn channel_pair() -> (Arc<ChannelTransport>, Arc<ChannelTransport>) {
    let (a_tx, b_rx) = mpsc::channel::<Vec<u8>>(32);
    let (b_tx, a_rx) = mpsc::channel::<Vec<u8>>(32);
    (
        Arc::new(ChannelTransport { tx: a_tx, rx: tokio::sync::Mutex::new(a_rx) }),
        Arc::new(ChannelTransport { tx: b_tx, rx: tokio::sync::Mutex::new(b_rx) }),
    )
}

// ── Tool + Handler ──────────────────────────────────────────────────────

struct CalcTool;

#[async_trait]
impl AgentTool for CalcTool {
    fn name(&self) -> &str { "add" }
    fn description(&self) -> &str { "Add two numbers" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]})
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let a = params["a"].as_i64().unwrap_or(0);
        let b = params["b"].as_i64().unwrap_or(0);
        Ok(serde_json::json!({"sum": a + b}))
    }
}

struct TestHandler;

#[async_trait]
impl AgentHandler for TestHandler {
    async fn handle_session(
        &self,
        manifest: ToolManifest,
        executor: &dyn ToolExecutor,
    ) -> Result<String> {
        assert!(manifest.tools.iter().any(|t| t.name == "add"));

        let r1 = executor.call_tool("c1", "add", r#"{"a":10,"b":32}"#).await?;
        assert!(r1.success);
        let v1: serde_json::Value = serde_json::from_str(&r1.result_json)?;
        assert_eq!(v1["sum"], 42);

        let r2 = executor.call_tool("c2", "add", r#"{"a":100,"b":200}"#).await?;
        assert!(r2.success);
        let v2: serde_json::Value = serde_json::from_str(&r2.result_json)?;
        assert_eq!(v2["sum"], 300);

        Ok(format!("10+32={}, 100+200={}", v1["sum"], v2["sum"]))
    }
}

// ── Test ─────────────────────────────────────────────────────────────────

/// Worker + Server over pure channel transport, no gRPC.
/// Proves the entire stack is transport-agnostic.
#[tokio::test]
async fn serve_session_over_channels() {
    let (agent_transport, server_transport) = channel_pair();

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(CalcTool));

    let worker = tokio::spawn(async move {
        run_worker(agent_transport.as_ref(), &tools, "s1").await
    });

    let server = tokio::spawn(async move {
        let handler = TestHandler;
        serve_session(server_transport.as_ref(), &handler).await
    });

    let server_result = server.await.unwrap().unwrap();
    assert!(server_result.contains("42"));
    assert!(server_result.contains("300"));

    let worker_result = worker.await.unwrap().unwrap();
    assert_eq!(worker_result.tool_calls_executed, 2);
    assert!(worker_result.final_text.contains("42"));

    eprintln!("server: {}", server_result);
    eprintln!("worker: {} calls, final={}", worker_result.tool_calls_executed, worker_result.final_text);
    eprintln!("=== TRANSPORT-AGNOSTIC SERVER TEST PASSED ===");
}
