//! Real LLM ReAct agent E2E tests.
//! Calls the actual DeepSeek API, executes real tools, multi-turn reasoning.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use cyber_agent_model::{LlmProvider, UserContent};
use cyber_agent_provider::BridgeProvider;
use cyber_agent_runner::{run_agent_loop, RunnerEvent};
use cyber_agent_tool::{AgentTool, ToolRegistry};
use cyber_agent_transport::Transport;

// ── HTTP Transport ──────────────────────────────────────────────────────

struct HttpTransport {
    endpoint: String,
    api_key: String,
}

#[async_trait]
impl Transport for HttpTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        let resp = ureq::post(&self.endpoint)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_bytes(data);

        match resp {
            Ok(r) => {
                let body = r.into_string()?;
                let wrapped = format!(r#"{{"payload":{}}}"#, body);
                let mut val: serde_json::Value = serde_json::from_str(&wrapped)?;
                normalize_nulls(&mut val);
                Ok(serde_json::to_vec(&val)?)
            }
            Err(ureq::Error::Status(code, r)) => {
                let body = r.into_string().unwrap_or_default();
                anyhow::bail!("API error (HTTP {}): {}", code, body)
            }
            Err(e) => anyhow::bail!("transport error: {}", e),
        }
    }
}

fn normalize_nulls(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                if val.is_null() && (key == "tool_calls" || key == "tools") {
                    *val = serde_json::Value::Array(vec![]);
                } else {
                    normalize_nulls(val);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                normalize_nulls(item);
            }
        }
        _ => {}
    }
}

// ── Tools ───────────────────────────────────────────────────────────────

struct ShellTool;

#[async_trait]
impl AgentTool for ShellTool {
    fn name(&self) -> &str { "shell" }
    fn description(&self) -> &str { "Execute a shell command and return stdout/stderr" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute" }
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let cmd = params.get("command").and_then(|v| v.as_str()).unwrap_or("echo noop");
        eprintln!("    [TOOL] $ {}", cmd);
        let out = std::process::Command::new("sh").args(["-c", cmd]).output()?;
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let mut r = String::new();
        if !stdout.is_empty() { r.push_str(&stdout); }
        if !stderr.is_empty() { if !r.is_empty() { r.push('\n'); } r.push_str("stderr: "); r.push_str(&stderr); }
        if r.is_empty() { r.push_str("(no output)"); }
        eprintln!("    [OUT]  {}", r.trim());
        Ok(serde_json::Value::String(r))
    }
}

struct ReadFileTool;

#[async_trait]
impl AgentTool for ReadFileTool {
    fn name(&self) -> &str { "read_file" }
    fn description(&self) -> &str { "Read file contents" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string", "description": "File path" } },
            "required": ["path"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
        eprintln!("    [TOOL] read_file: {}", path);
        match std::fs::read_to_string(path) {
            Ok(c) => { eprintln!("    [OUT]  {} bytes", c.len()); Ok(serde_json::json!({"content": c, "size": c.len()})) }
            Err(e) => { eprintln!("    [ERR]  {}", e); Ok(serde_json::json!({"error": e.to_string()})) }
        }
    }
}

struct WriteFileTool;

#[async_trait]
impl AgentTool for WriteFileTool {
    fn name(&self) -> &str { "write_file" }
    fn description(&self) -> &str { "Write content to a file" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
        eprintln!("    [TOOL] write_file: {} ({} bytes)", path, content.len());
        if let Some(p) = std::path::Path::new(path).parent() {
            if !p.as_os_str().is_empty() { let _ = std::fs::create_dir_all(p); }
        }
        std::fs::write(path, content)?;
        eprintln!("    [OUT]  written");
        Ok(serde_json::json!({"path": path, "bytes_written": content.len()}))
    }
}

// ── Config ──────────────────────────────────────────────────────────────

fn config() -> Option<(String, String, String)> {
    let key = std::env::var("CYBER_AGENT_API_KEY").unwrap_or_default();
    if key.is_empty() { return None; }
    let base = std::env::var("CYBER_AGENT_BASE_URL").unwrap_or("https://api.deepseek.com".into());
    let model = std::env::var("CYBER_AGENT_MODEL").unwrap_or("deepseek-chat".into());
    Some((format!("{}/v1/chat/completions", base.trim_end_matches('/')), key, model))
}

fn make_provider(endpoint: &str, api_key: &str, model: &str) -> Arc<dyn LlmProvider> {
    let transport = Arc::new(HttpTransport {
        endpoint: endpoint.to_string(),
        api_key: api_key.to_string(),
    });
    Arc::new(BridgeProvider::new(endpoint.into(), model.into(), "deepseek".into(), transport))
}

fn event_logger() -> Box<dyn Fn(RunnerEvent) + Send + Sync> {
    Box::new(|event| match &event {
        RunnerEvent::Iteration(n) => eprintln!("\n  ── Iteration {} ──", n),
        RunnerEvent::ToolCallStart { name, arguments, .. } =>
            eprintln!("  [LLM→TOOL] {} {}", name, arguments),
        RunnerEvent::ToolCallEnd { name, success, error, .. } =>
            eprintln!("  [TOOL→LLM] {} ok={} err={:?}", name, success, error),
    })
}

// ── Test 1: System info gathering ───────────────────────────────────────

#[tokio::test]
async fn real_agent_sysinfo() {
    let Some((endpoint, key, model)) = config() else {
        eprintln!("SKIP: set CYBER_AGENT_API_KEY"); return;
    };
    eprintln!("\n  ====== REAL AGENT: sysinfo ({}) ======\n", model);

    let provider = make_provider(&endpoint, &key, &model);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));
    tools.register(Box::new(ReadFileTool));
    let cb = event_logger();

    let result = run_agent_loop(
        provider, &tools,
        "You are a system recon assistant on Linux. Use shell and read_file tools. Be efficient, batch commands. Give a structured report.",
        &UserContent::Text(
            "Gather and report:\n1. hostname and user (hostname && whoami)\n2. kernel (uname -r)\n3. OS (cat /etc/os-release | head -4)\n4. uptime".into(),
        ),
        Some(&cb), None,
    ).await.expect("agent should succeed");

    eprintln!("\n  ====== REPORT ======");
    eprintln!("{}", result.text);
    eprintln!("  ====== iterations={} tool_calls={} tokens=in:{}/out:{} ======\n",
        result.iterations, result.tool_calls_made, result.usage.input_tokens, result.usage.output_tokens);

    assert!(!result.text.is_empty());
    assert!(result.tool_calls_made >= 1, "expected tool calls, got {}", result.tool_calls_made);
}

// ── Test 2: File write → read → verify ──────────────────────────────────

#[tokio::test]
async fn real_agent_file_task() {
    let Some((endpoint, key, model)) = config() else {
        eprintln!("SKIP: set CYBER_AGENT_API_KEY"); return;
    };
    eprintln!("\n  ====== REAL AGENT: file task ({}) ======\n", model);

    let provider = make_provider(&endpoint, &key, &model);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));
    tools.register(Box::new(ReadFileTool));
    tools.register(Box::new(WriteFileTool));
    let cb = event_logger();

    let test_dir = "/tmp/cyber_agent_e2e";
    let _ = std::fs::create_dir_all(test_dir);

    let result = run_agent_loop(
        provider, &tools,
        "You are a helpful assistant with shell, read_file, write_file tools. Complete each step and verify.",
        &UserContent::Text(format!(
            "Steps:\n\
             1. write_file: create '{test_dir}/hello.txt' with 'Hello from cyber-agent!'\n\
             2. read_file: read it back, confirm content\n\
             3. shell: run 'wc -c {test_dir}/hello.txt'\n\
             4. Report success/failure"
        ).into()),
        Some(&cb), None,
    ).await.expect("agent should succeed");

    eprintln!("\n  ====== REPORT ======");
    eprintln!("{}", result.text);
    eprintln!("  ====== iterations={} tool_calls={} ======\n",
        result.iterations, result.tool_calls_made);

    assert!(!result.text.is_empty());
    assert!(result.tool_calls_made >= 3, "expected >=3 tool calls, got {}", result.tool_calls_made);

    // Verify file was actually created by the agent
    let content = std::fs::read_to_string(format!("{}/hello.txt", test_dir)).expect("file should exist");
    assert_eq!(content, "Hello from cyber-agent!");
    eprintln!("  [VERIFY] file content = {:?} ✓", content);

    let _ = std::fs::remove_dir_all(test_dir);
}
