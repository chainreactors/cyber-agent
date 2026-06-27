//! E2E test with a local HTTP server simulating an OpenAI-compatible LLM API.
//!
//! This validates the full round-trip: HttpTransport → real HTTP → JSON serde →
//! BridgeProvider → Runner → Tool execution → multi-turn conversation.
//!
//! Unlike integration.rs (which uses a mock Transport), this test sends real
//! HTTP requests over TCP, testing the actual wire protocol byte-for-byte.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::Result;
use async_trait::async_trait;

use cyber_agent_proto::LlmProvider;
use cyber_agent_provider::BridgeProvider;
use cyber_agent_runner::{run_agent_loop, RunnerEvent};
use cyber_agent_tool::{AgentTool, ToolRegistry};
use cyber_agent_proto::Transport;

// ── HTTP Transport (same as what a real deployment would use) ────────────

struct HttpTransport {
    url: String,
}

#[async_trait]
impl Transport for HttpTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        let resp = ureq::post(&self.url)
            .set("Content-Type", "application/json")
            .send_bytes(data);

        match resp {
            Ok(r) => {
                let body = r.into_string()?;
                // Wrap the OpenAI response in BridgeResponse envelope
                let wrapped = format!(r#"{{"payload":{}}}"#, body);
                Ok(wrapped.into_bytes())
            }
            Err(ureq::Error::Status(code, r)) => {
                let body = r.into_string().unwrap_or_default();
                anyhow::bail!("HTTP {}: {}", code, body)
            }
            Err(e) => anyhow::bail!("transport: {}", e),
        }
    }
}

// ── Local HTTP Server simulating OpenAI Chat Completions API ────────────

fn start_mock_llm_server() -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", addr.port());

    let handle = thread::spawn(move || {
        let call_count = AtomicUsize::new(0);

        // Handle 3 requests then stop
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => break,
            };

            // Read HTTP request
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();

            // Read headers to find Content-Length
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if line.to_lowercase().starts_with("content-length:") {
                    content_length = line.split(':').nth(1).unwrap().trim().parse().unwrap();
                }
            }

            // Read body
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body).unwrap();

            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let n = call_count.fetch_add(1, Ordering::SeqCst);

            eprintln!(
                "  [server] request #{}: model={}, messages={}, tools={}",
                n,
                request["model"].as_str().unwrap_or("?"),
                request["messages"].as_array().map(|a| a.len()).unwrap_or(0),
                request["tools"].as_array().map(|a| a.len()).unwrap_or(0),
            );

            // Generate response based on call number
            let response_body = match n {
                0 => {
                    // First call: return a tool call
                    serde_json::json!({
                        "choices": [{
                            "message": {
                                "content": "Let me check that for you.",
                                "tool_calls": [{
                                    "id": "call_real_001",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": "{\"command\":\"echo hello_from_server\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20
                        }
                    })
                }
                1 => {
                    // Second call: verify tool result was sent, return final answer
                    let messages = request["messages"].as_array().unwrap();
                    let tool_msg = messages.iter().find(|m| m["role"] == "tool");
                    assert!(tool_msg.is_some(), "expected tool result in second request");
                    let content = tool_msg.unwrap()["content"].as_str().unwrap_or("");
                    eprintln!("  [server] tool result content: {}", &content[..content.len().min(200)]);

                    serde_json::json!({
                        "choices": [{
                            "message": {
                                "content": "The command output: hello_from_server",
                                "tool_calls": []
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 200,
                            "completion_tokens": 15
                        }
                    })
                }
                _ => {
                    serde_json::json!({
                        "choices": [{
                            "message": {
                                "content": "done",
                                "tool_calls": []
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                    })
                }
            };

            let body_str = serde_json::to_string(&response_body).unwrap();
            let http_response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            stream.write_all(http_response.as_bytes()).unwrap();
            stream.flush().unwrap();

            // Stop after the final answer (no more tool calls)
            if n >= 1 {
                // Accept one more connection attempt then drop the listener
                drop(listener);
                break;
            }
        }
    });

    (url, handle)
}

// ── Test Tool ───────────────────────────────────────────────────────────

struct ShellTool;

#[async_trait]
impl AgentTool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }
    fn description(&self) -> &str {
        "Execute a shell command"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"}
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let command = params.get("command").and_then(|v| v.as_str()).unwrap_or("echo noop");
        eprintln!("  [tool] executing: {}", command);

        let output = std::process::Command::new("sh")
            .args(["-c", command])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        eprintln!("  [tool] output: {}", stdout);
        Ok(serde_json::Value::String(stdout))
    }
}

// ── E2E Test ────────────────────────────────────────────────────────────

/// Full E2E test over real HTTP:
///   HttpTransport → TCP → local mock server → BridgeProvider → Runner → ShellTool
///
/// This proves the complete pipeline works end-to-end with real network I/O,
/// real JSON serialization, and real tool execution.
#[tokio::test]
async fn e2e_over_http_with_tool_call() {
    eprintln!("\n=== E2E: local HTTP server with tool calling ===\n");

    // Start local mock LLM server
    let (url, server_handle) = start_mock_llm_server();
    eprintln!("[setup] mock server at: {}", url);

    // Build the full stack
    let transport = Arc::new(HttpTransport { url: url.clone() });
    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        url,
        "test-model".into(),
        "local-mock".into(),
        transport,
    ));

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));

    // Track events
    let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let on_event: Box<dyn Fn(RunnerEvent) + Send + Sync> = Box::new(move |event| {
        let msg = match &event {
            RunnerEvent::Iteration(n) => format!("iteration:{}", n),
            RunnerEvent::ToolCallStart { name, .. } => format!("tool_start:{}", name),
            RunnerEvent::ToolCallEnd { name, success, .. } => {
                format!("tool_end:{}:{}", name, success)
            }
        };
        eprintln!("[event] {}", msg);
        events_clone.lock().unwrap().push(msg);
    });

    // Run the agent loop
    let result = run_agent_loop(
        provider,
        &tools,
        "You are a test assistant.",
        "Run echo hello_from_server",
        Some(&on_event),
        None,
    )
    .await
    .expect("agent loop should succeed");

    // Wait for server thread
    server_handle.join().expect("server thread should not panic");

    // ── Assertions ──────────────────────────────────────────────────────

    eprintln!("\n[result] text: {}", result.text);
    eprintln!(
        "[result] iterations={}, tool_calls={}, usage={{in={}, out={}}}",
        result.iterations,
        result.tool_calls_made,
        result.usage.input_tokens,
        result.usage.output_tokens
    );

    // The final text should contain the server's response
    assert!(
        result.text.contains("hello_from_server"),
        "expected 'hello_from_server' in result: {}",
        result.text
    );

    // Should have 2 iterations (tool_call + final_text)
    assert_eq!(result.iterations, 2);

    // Should have made 1 tool call
    assert_eq!(result.tool_calls_made, 1);

    // Usage should be accumulated
    assert_eq!(result.usage.input_tokens, 300); // 100 + 200
    assert_eq!(result.usage.output_tokens, 35); // 20 + 15

    // Check events
    let events = events.lock().unwrap();
    assert!(events.contains(&"iteration:1".to_string()));
    assert!(events.contains(&"iteration:2".to_string()));
    assert!(events.contains(&"tool_start:shell".to_string()));
    assert!(events.contains(&"tool_end:shell:true".to_string()));

    eprintln!("\n=== E2E OVER HTTP: ALL ASSERTIONS PASSED ===\n");
}
