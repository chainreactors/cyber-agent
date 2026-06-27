use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use cyber_agent_proto::LlmProvider;
use cyber_agent_provider::BridgeProvider;
use cyber_agent_runner::{run_agent_loop, RunnerEvent};
use cyber_agent_tool::{AgentTool, ToolRegistry};
use cyber_agent_transport::Transport;

// ── HTTP Transport: forwards raw BridgeRequest JSON to a real OpenAI-compatible API ──

struct HttpLlmTransport {
    endpoint: String,
    api_key: String,
}

#[async_trait]
impl Transport for HttpLlmTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        // data is a serialized BridgeRequest (OpenAI-compatible JSON)
        // Forward it directly to the LLM API endpoint
        let response = ureq::post(&self.endpoint)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_bytes(data);

        match response {
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                // Wrap in BridgeResponse format: {"payload": <api_response>}
                // The API returns an OpenAI-compatible completion response directly,
                // which matches BridgeCompletionPayload
                let wrapped = format!(r#"{{"payload":{}}}"#, body);

                // Normalize null tool_calls to empty array (some APIs return null)
                let mut value: serde_json::Value = serde_json::from_str(&wrapped)?;
                normalize_null_arrays(&mut value);
                Ok(serde_json::to_vec(&value)?)
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                anyhow::bail!("API error (HTTP {}): {}", code, body)
            }
            Err(e) => anyhow::bail!("transport error: {}", e),
        }
    }
}

fn normalize_null_arrays(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                if val.is_null() && (key == "tool_calls" || key == "tools") {
                    *val = serde_json::Value::Array(vec![]);
                } else {
                    normalize_null_arrays(val);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                normalize_null_arrays(item);
            }
        }
        _ => {}
    }
}

// ── Tools ───────────────────────────────────────────────────────────────

struct ShellTool;

#[async_trait]
impl AgentTool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }
    fn description(&self) -> &str {
        "Execute a shell command on the host and return stdout/stderr"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("echo noop");

        eprintln!("  [tool] shell: {}", command);

        let output = std::process::Command::new("sh")
            .args(["-c", command])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().unwrap_or(-1);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("stderr: ");
            result.push_str(&stderr);
        }
        if result.is_empty() {
            result.push_str("(no output)");
        }
        if code != 0 {
            result.push_str(&format!("\nexit_code: {}", code));
        }

        eprintln!("  [result] ({} chars) {}", result.len(), result.trim());
        Ok(serde_json::Value::String(result))
    }
}

struct ListDirTool;

#[async_trait]
impl AgentTool for ListDirTool {
    fn name(&self) -> &str {
        "list_directory"
    }
    fn description(&self) -> &str {
        "List files and directories at a given path"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list"
                }
            },
            "required": ["path"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        eprintln!("  [tool] list_directory: {}", path);
        match std::fs::read_dir(path) {
            Ok(entries) => {
                let items: Vec<serde_json::Value> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let meta = e.metadata().ok();
                        serde_json::json!({
                            "name": e.file_name().to_string_lossy(),
                            "is_dir": meta.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                            "size": meta.as_ref().map(|m| m.len()).unwrap_or(0),
                        })
                    })
                    .collect();
                eprintln!("  [result] {} entries", items.len());
                Ok(serde_json::json!({"path": path, "entries": items}))
            }
            Err(e) => Ok(serde_json::json!({"error": format!("{}", e)})),
        }
    }
}

// ── E2E Tests ───────────────────────────────────────────────────────────

fn get_config() -> (String, String, String) {
    let base_url = std::env::var("CYBER_AGENT_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    let api_key = std::env::var("CYBER_AGENT_API_KEY").unwrap_or_default();
    let model = std::env::var("CYBER_AGENT_MODEL")
        .unwrap_or_else(|_| "deepseek-chat".to_string());

    let endpoint = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    (endpoint, api_key, model)
}

/// E2E test: simple text completion through the full stack with a real LLM
#[tokio::test]
async fn e2e_simple_completion() {
    let (endpoint, api_key, model) = get_config();
    if api_key.is_empty() {
        eprintln!("SKIPPING: set CYBER_AGENT_API_KEY to run E2E tests");
        return;
    }

    eprintln!("\n=== E2E: simple completion ===");
    eprintln!("[config] endpoint={}, model={}", endpoint, model);

    let transport = Arc::new(HttpLlmTransport {
        endpoint: endpoint.clone(),
        api_key: api_key.clone(),
    });
    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        endpoint,
        model,
        "deepseek".into(),
        transport,
    ));

    let result = run_agent_loop(
        provider,
        &ToolRegistry::new(),
        "You are a helpful assistant. Be very brief.",
        "Say 'cyber_agent_ok' and nothing else.",
        None,
        None,
    )
    .await
    .expect("completion should succeed");

    eprintln!("[result] text: {}", result.text);
    eprintln!(
        "[result] iterations={}, usage={{in={}, out={}}}",
        result.iterations, result.usage.input_tokens, result.usage.output_tokens
    );
    assert!(!result.text.is_empty(), "expected non-empty response");
    assert_eq!(result.iterations, 1);
    assert_eq!(result.tool_calls_made, 0);
    eprintln!("=== PASSED ===\n");
}

/// E2E test: agent uses shell tool to execute a real command
#[tokio::test]
async fn e2e_tool_call() {
    let (endpoint, api_key, model) = get_config();
    if api_key.is_empty() {
        eprintln!("SKIPPING: set CYBER_AGENT_API_KEY to run E2E tests");
        return;
    }

    eprintln!("\n=== E2E: tool call ===");
    eprintln!("[config] endpoint={}, model={}", endpoint, model);

    let transport = Arc::new(HttpLlmTransport {
        endpoint: endpoint.clone(),
        api_key: api_key.clone(),
    });
    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        endpoint,
        model,
        "deepseek".into(),
        transport,
    ));

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));

    let on_event: Box<dyn Fn(RunnerEvent) + Send + Sync> = Box::new(|event| match event {
        RunnerEvent::Iteration(n) => eprintln!("[event] iteration {}", n),
        RunnerEvent::ToolCallStart { name, .. } => eprintln!("[event] tool_call_start: {}", name),
        RunnerEvent::ToolCallEnd {
            name, success, error, ..
        } => eprintln!(
            "[event] tool_call_end: {} success={} error={:?}",
            name, success, error
        ),
    });

    let result = run_agent_loop(
        provider,
        &tools,
        "You are a helpful assistant. Use the shell tool when asked to run commands. Be brief.",
        "Run the command 'echo hello_from_cyber_agent' and tell me what it outputs.",
        Some(&on_event),
        None,
    )
    .await
    .expect("agent loop should succeed");

    eprintln!("[result] text: {}", result.text);
    eprintln!(
        "[result] iterations={}, tool_calls={}, usage={{in={}, out={}}}",
        result.iterations,
        result.tool_calls_made,
        result.usage.input_tokens,
        result.usage.output_tokens
    );

    assert!(!result.text.is_empty(), "expected non-empty response");
    assert!(
        result.tool_calls_made >= 1,
        "expected at least 1 tool call, got {}",
        result.tool_calls_made
    );
    eprintln!("=== PASSED ===\n");
}

/// E2E test: agent uses multiple tools in one session
#[tokio::test]
async fn e2e_multi_tool() {
    let (endpoint, api_key, model) = get_config();
    if api_key.is_empty() {
        eprintln!("SKIPPING: set CYBER_AGENT_API_KEY to run E2E tests");
        return;
    }

    eprintln!("\n=== E2E: multi-tool ===");

    let transport = Arc::new(HttpLlmTransport {
        endpoint: endpoint.clone(),
        api_key: api_key.clone(),
    });
    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        endpoint,
        model,
        "deepseek".into(),
        transport,
    ));

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));
    tools.register(Box::new(ListDirTool));

    let result = run_agent_loop(
        provider,
        &tools,
        "You are a system assistant with shell and list_directory tools. Use the appropriate tool for each task. Be concise.",
        "Use the list_directory tool to list files in /tmp, then use shell to show the current date.",
        None,
        None,
    )
    .await
    .expect("agent loop should succeed");

    eprintln!("[result] text: {}", result.text);
    eprintln!(
        "[result] iterations={}, tool_calls={}",
        result.iterations, result.tool_calls_made
    );

    assert!(!result.text.is_empty());
    assert!(
        result.tool_calls_made >= 2,
        "expected at least 2 tool calls, got {}",
        result.tool_calls_made
    );
    eprintln!("=== PASSED ===\n");
}
