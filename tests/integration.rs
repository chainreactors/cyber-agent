use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;

use cyber_agent_proto::LlmProvider;
use cyber_agent_provider::wire::{
    BridgeAssistantMessage, BridgeChoice, BridgeCompletionPayload, BridgeRequest, BridgeResponse,
    BridgeResponsePayload, BridgeToolCall, BridgeToolCallFunction, BridgeUsage,
};
use cyber_agent_provider::BridgeProvider;
use cyber_agent_runner::{run_agent_loop, RunnerEvent};
use cyber_agent_tool::{AgentTool, ToolRegistry};
use cyber_agent_proto::Transport;

// ── Simulated LLM Backend ───────────────────────────────────────────────
// This simulates what happens on the server side: receives a BridgeRequest,
// inspects the conversation history and available tools, decides whether to
// make a tool call or return a final text answer.

struct SimulatedLlmBackend {
    call_count: AtomicUsize,
    requests: Mutex<Vec<BridgeRequest>>,
}

impl SimulatedLlmBackend {
    fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for SimulatedLlmBackend {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Deserialize the request — just like a real server would
        let req: BridgeRequest = serde_json::from_slice(data)?;
        self.requests.lock().unwrap().push(req.clone());

        let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);

        let response = match call_num {
            0 => {
                // First call: the LLM decides to use the shell tool
                assert!(
                    !req.tools.is_empty(),
                    "expected tools to be sent to the LLM"
                );
                let has_shell = req
                    .tools
                    .iter()
                    .any(|t| t.function.name == "shell");
                assert!(has_shell, "expected 'shell' tool in the request");

                BridgeResponse {
                    payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                        choices: vec![BridgeChoice {
                            message: BridgeAssistantMessage {
                                content: Some(
                                    "I'll run the hostname command for you.".to_string(),
                                ),
                                tool_calls: vec![BridgeToolCall {
                                    id: "call_001".to_string(),
                                    call_type: "function".to_string(),
                                    function: BridgeToolCallFunction {
                                        name: "shell".to_string(),
                                        arguments: r#"{"command":"hostname"}"#.to_string(),
                                    },
                                }],
                            },
                            finish_reason: Some("tool_calls".to_string()),
                        }],
                        usage: BridgeUsage {
                            prompt_tokens: 150,
                            completion_tokens: 30,
                        },
                    }),
                }
            }
            1 => {
                // Second call: the LLM has received the tool result, now it
                // should also call list_directory
                let last_msg = req.messages.last().unwrap();
                assert_eq!(last_msg["role"], "tool", "expected tool result message");
                let tool_output = last_msg["content"].as_str().unwrap_or("");
                assert!(
                    !tool_output.is_empty(),
                    "expected non-empty tool result, got empty"
                );

                BridgeResponse {
                    payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                        choices: vec![BridgeChoice {
                            message: BridgeAssistantMessage {
                                content: Some("Now let me list the current directory.".to_string()),
                                tool_calls: vec![BridgeToolCall {
                                    id: "call_002".to_string(),
                                    call_type: "function".to_string(),
                                    function: BridgeToolCallFunction {
                                        name: "list_directory".to_string(),
                                        arguments: r#"{"path":"."}"#.to_string(),
                                    },
                                }],
                            },
                            finish_reason: Some("tool_calls".to_string()),
                        }],
                        usage: BridgeUsage {
                            prompt_tokens: 200,
                            completion_tokens: 25,
                        },
                    }),
                }
            }
            _ => {
                // Third call: the LLM returns a final text answer
                BridgeResponse {
                    payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                        choices: vec![BridgeChoice {
                            message: BridgeAssistantMessage {
                                content: Some(
                                    "The hostname is test-host and the current directory contains some files."
                                        .to_string(),
                                ),
                                tool_calls: vec![],
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        usage: BridgeUsage {
                            prompt_tokens: 300,
                            completion_tokens: 20,
                        },
                    }),
                }
            }
        };

        // Serialize response — just like a real server would
        Ok(serde_json::to_vec(&response)?)
    }
}

// ── Test Tools ──────────────────────────────────────────────────────────

struct ShellTool;

#[async_trait]
impl AgentTool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }
    fn description(&self) -> &str {
        "Execute a shell command on the target host"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command to execute" }
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Actually execute the command
        let output = std::process::Command::new("sh")
            .args(["-c", command])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);

        Ok(serde_json::json!({
            "stdout": stdout.trim(),
            "stderr": stderr.trim(),
            "exit_code": code,
        }))
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
                "path": { "type": "string", "description": "Directory path to list" }
            },
            "required": ["path"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let entries: Vec<String> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        Ok(serde_json::json!({ "path": path, "entries": entries }))
    }
}

// ── Integration Tests ───────────────────────────────────────────────────

/// Full stack test: Transport → Protocol → Provider → Runner → Tools → loop
///
/// This simulates the complete malefic agent flow:
/// 1. User asks "what is the hostname and list current directory"
/// 2. Simulated LLM backend decides to call "shell" tool with "hostname"
/// 3. Runner executes the shell tool, gets real output
/// 4. Tool result is sent back through the bridge
/// 5. LLM decides to call "list_directory" tool
/// 6. Runner executes list_directory, gets real output
/// 7. LLM returns final text answer
#[tokio::test]
async fn full_stack_multi_tool_agent_loop() {
    let backend = Arc::new(SimulatedLlmBackend::new());

    // Build the provider from transport (just like malefic-3rd does)
    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        "test://simulated".into(),
        "gpt-4o".into(),
        "test-bridge".into(),
        backend.clone(),
    ));

    // Register tools
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ShellTool));
    tools.register(Box::new(ListDirTool));

    // Track events
    let events: Arc<Mutex<Vec<RunnerEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let on_event: Box<dyn Fn(RunnerEvent) + Send + Sync> = Box::new(move |event| {
        events_clone.lock().unwrap().push(event);
    });

    // Run the agent loop
    let result = run_agent_loop(
        provider,
        &tools,
        "You are a helpful assistant with shell and directory listing tools.",
        "What is the hostname of this machine? Also list the current directory.",
        Some(&on_event),
        None,
    )
    .await
    .expect("agent loop should succeed");

    // ── Verify results ──────────────────────────────────────────────────

    // The agent should have completed with a final text answer
    assert!(
        !result.text.is_empty(),
        "expected non-empty final text, got empty"
    );
    assert!(
        result.text.contains("hostname"),
        "final text should mention hostname: {}",
        result.text
    );
    eprintln!("[result] text: {}", result.text);

    // 3 iterations: tool_call → tool_call → final_text
    assert_eq!(result.iterations, 3, "expected 3 iterations");

    // 2 tool calls total (shell + list_directory)
    assert_eq!(result.tool_calls_made, 2, "expected 2 tool calls");

    // Usage should be accumulated across all 3 LLM calls
    assert_eq!(
        result.usage.input_tokens,
        150 + 200 + 300,
        "input tokens should be sum of all calls"
    );
    assert_eq!(
        result.usage.output_tokens,
        30 + 25 + 20,
        "output tokens should be sum of all calls"
    );
    eprintln!(
        "[result] iterations={}, tool_calls={}, usage={{in={}, out={}}}",
        result.iterations,
        result.tool_calls_made,
        result.usage.input_tokens,
        result.usage.output_tokens,
    );

    // ── Verify the requests sent to the backend ─────────────────────────

    let requests = backend.requests.lock().unwrap();
    assert_eq!(requests.len(), 3, "backend should have received 3 requests");

    // First request: system + user messages, with tool schemas
    assert_eq!(requests[0].model, "gpt-4o");
    assert_eq!(requests[0].messages.len(), 2); // system + user
    assert_eq!(requests[0].messages[0]["role"], "system");
    assert_eq!(requests[0].messages[1]["role"], "user");
    assert_eq!(requests[0].tools.len(), 2); // shell + list_directory
    eprintln!(
        "[req 0] messages={}, tools={}",
        requests[0].messages.len(),
        requests[0].tools.len()
    );

    // Second request: system + user + assistant(tool_call) + tool(result)
    assert_eq!(requests[1].messages.len(), 4);
    assert_eq!(requests[1].messages[2]["role"], "assistant");
    assert_eq!(requests[1].messages[3]["role"], "tool");
    assert_eq!(requests[1].messages[3]["tool_call_id"], "call_001");
    eprintln!(
        "[req 1] messages={}, tool_result_role={}",
        requests[1].messages.len(),
        requests[1].messages[3]["role"]
    );

    // Third request: system + user + assistant + tool + assistant + tool
    assert_eq!(requests[2].messages.len(), 6);
    assert_eq!(requests[2].messages[5]["tool_call_id"], "call_002");
    eprintln!(
        "[req 2] messages={}, final tool_call_id={}",
        requests[2].messages.len(),
        requests[2].messages[5]["tool_call_id"]
    );

    // ── Verify events ───────────────────────────────────────────────────

    let events = events.lock().unwrap();
    let iterations: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, RunnerEvent::Iteration(_)))
        .collect();
    assert_eq!(iterations.len(), 3, "expected 3 Iteration events");

    let tool_starts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, RunnerEvent::ToolCallStart { .. }))
        .collect();
    assert_eq!(tool_starts.len(), 2, "expected 2 ToolCallStart events");

    let tool_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, RunnerEvent::ToolCallEnd { .. }))
        .collect();
    assert_eq!(tool_ends.len(), 2, "expected 2 ToolCallEnd events");

    // Both tool calls should have succeeded
    for event in &tool_ends {
        if let RunnerEvent::ToolCallEnd { success, name, .. } = event {
            assert!(success, "tool '{}' should have succeeded", name);
        }
    }

    eprintln!("[events] total={}, iterations=3, tool_starts=2, tool_ends=2", events.len());
    eprintln!("\n=== FULL STACK TEST PASSED ===");
}

/// Test: the runner correctly handles a tool that returns an error
#[tokio::test]
async fn tool_error_is_forwarded_to_llm() {
    struct ErrorOnFirstCallBackend {
        call_count: AtomicUsize,
    }

    #[async_trait]
    impl Transport for ErrorOnFirstCallBackend {
        async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
            let req: BridgeRequest = serde_json::from_slice(data)?;
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);

            let response = if n == 0 {
                // Ask the agent to call a tool that doesn't exist
                BridgeResponse {
                    payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                        choices: vec![BridgeChoice {
                            message: BridgeAssistantMessage {
                                content: None,
                                tool_calls: vec![BridgeToolCall {
                                    id: "call_err".into(),
                                    call_type: "function".into(),
                                    function: BridgeToolCallFunction {
                                        name: "nonexistent_tool".into(),
                                        arguments: "{}".into(),
                                    },
                                }],
                            },
                            finish_reason: Some("tool_calls".into()),
                        }],
                        usage: BridgeUsage::default(),
                    }),
                }
            } else {
                // After seeing the error, verify the tool result contains the error
                let tool_msg = req.messages.iter().find(|m| m["role"] == "tool").unwrap();
                let content = tool_msg["content"].as_str().unwrap();
                assert!(
                    content.contains("unknown tool"),
                    "tool error should be forwarded: {}",
                    content
                );

                BridgeResponse {
                    payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                        choices: vec![BridgeChoice {
                            message: BridgeAssistantMessage {
                                content: Some("Sorry, that tool is not available.".into()),
                                tool_calls: vec![],
                            },
                            finish_reason: Some("stop".into()),
                        }],
                        usage: BridgeUsage::default(),
                    }),
                }
            };

            Ok(serde_json::to_vec(&response)?)
        }
    }

    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        "test://".into(),
        "model".into(),
        "test".into(),
        Arc::new(ErrorOnFirstCallBackend {
            call_count: AtomicUsize::new(0),
        }),
    ));

    let tools = ToolRegistry::new(); // empty — no tools registered

    let result = run_agent_loop(
        provider,
        &tools,
        "system",
        "test",
        None,
        None,
    )
    .await
    .expect("should complete despite tool error");

    assert!(result.text.contains("not available"));
    assert_eq!(result.tool_calls_made, 1);
    eprintln!("[error test] text: {}", result.text);
    eprintln!("=== TOOL ERROR TEST PASSED ===");
}

/// Test: conversation history is correctly forwarded through the bridge
#[tokio::test]
async fn history_is_forwarded() {
    use cyber_agent_proto::ChatMessage;

    struct HistoryCheckBackend;

    #[async_trait]
    impl Transport for HistoryCheckBackend {
        async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
            let req: BridgeRequest = serde_json::from_slice(data)?;

            // Should have: system + history_user + history_assistant + current_user = 4 messages
            assert_eq!(
                req.messages.len(),
                4,
                "expected 4 messages (system + 2 history + user), got {}",
                req.messages.len()
            );
            assert_eq!(req.messages[0]["role"], "system");
            assert_eq!(req.messages[1]["role"], "user");
            assert_eq!(req.messages[1]["content"], "previous question");
            assert_eq!(req.messages[2]["role"], "assistant");
            assert_eq!(req.messages[2]["content"], "previous answer");
            assert_eq!(req.messages[3]["role"], "user");
            assert_eq!(req.messages[3]["content"], "follow-up question");

            let response = BridgeResponse {
                payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                    choices: vec![BridgeChoice {
                        message: BridgeAssistantMessage {
                            content: Some("follow-up answer".into()),
                            tool_calls: vec![],
                        },
                        finish_reason: Some("stop".into()),
                    }],
                    usage: BridgeUsage::default(),
                }),
            };
            Ok(serde_json::to_vec(&response)?)
        }
    }

    let provider: Arc<dyn LlmProvider> = Arc::new(BridgeProvider::new(
        "test://".into(),
        "model".into(),
        "test".into(),
        Arc::new(HistoryCheckBackend),
    ));

    let history = vec![
        ChatMessage::user("previous question"),
        ChatMessage::assistant("previous answer"),
    ];

    let result = run_agent_loop(
        provider,
        &ToolRegistry::new(),
        "system prompt",
        "follow-up question",
        None,
        Some(history),
    )
    .await
    .expect("should succeed");

    assert_eq!(result.text, "follow-up answer");
    eprintln!("[history test] text: {}", result.text);
    eprintln!("=== HISTORY TEST PASSED ===");
}

/// Test: protocol serialization round-trip
#[test]
fn protocol_serde_round_trip() {
    use cyber_agent_provider::wire::*;

    // Build a request with tools
    let request = BridgeRequest {
        model: "gpt-4o".into(),
        messages: vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hello"}),
        ],
        tools: vec![BridgeRequestTool {
            tool_type: "function".into(),
            function: BridgeRequestFunction {
                name: "shell".into(),
                description: "Execute command".into(),
                parameters: serde_json::json!({"type": "object"}),
                strict: false,
            },
        }],
    };

    // Serialize to bytes and back
    let bytes = serde_json::to_vec(&request).unwrap();
    let deserialized: BridgeRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(deserialized.model, "gpt-4o");
    assert_eq!(deserialized.messages.len(), 2);
    assert_eq!(deserialized.tools.len(), 1);
    assert_eq!(deserialized.tools[0].function.name, "shell");

    // Build a response with tool calls
    let response = BridgeResponse {
        payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
            choices: vec![BridgeChoice {
                message: BridgeAssistantMessage {
                    content: Some("thinking".into()),
                    tool_calls: vec![BridgeToolCall {
                        id: "call_1".into(),
                        call_type: "function".into(),
                        function: BridgeToolCallFunction {
                            name: "shell".into(),
                            arguments: r#"{"command":"ls"}"#.into(),
                        },
                    }],
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: BridgeUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
            },
        }),
    };

    let resp_bytes = serde_json::to_vec(&response).unwrap();
    let deserialized: BridgeResponse = serde_json::from_slice(&resp_bytes).unwrap();
    match deserialized.payload {
        BridgeResponsePayload::Completion(payload) => {
            assert_eq!(payload.choices.len(), 1);
            assert_eq!(
                payload.choices[0].message.content.as_deref(),
                Some("thinking")
            );
            assert_eq!(payload.choices[0].message.tool_calls.len(), 1);
            assert_eq!(
                payload.choices[0].message.tool_calls[0].function.name,
                "shell"
            );
            assert_eq!(payload.usage.prompt_tokens, 100);
        }
        _ => panic!("expected completion payload"),
    }

    // Error response round-trip
    let error_response = BridgeResponse {
        payload: BridgeResponsePayload::Error(BridgeErrorPayload {
            error: BridgeErrorBody::Message {
                message: "rate limited".into(),
            },
        }),
    };
    let err_bytes = serde_json::to_vec(&error_response).unwrap();
    let deserialized: BridgeResponse = serde_json::from_slice(&err_bytes).unwrap();
    match deserialized.payload {
        BridgeResponsePayload::Error(err) => {
            assert_eq!(err.error.message().unwrap(), "rate limited");
        }
        _ => panic!("expected error payload"),
    }

    eprintln!("=== PROTOCOL ROUND-TRIP TEST PASSED ===");
}
