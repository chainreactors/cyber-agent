use std::fmt::Write;
use std::sync::Arc;

use anyhow::anyhow;

use cyber_agent_proto::{ChatMessage, CompletionResponse, LlmProvider, ToolDef, Usage};
use cyber_agent_tool::ToolRegistry;

const DEFAULT_MAX_ITERATIONS: usize = 25;
const MAX_TOOL_RESULT_BYTES: usize = 100_000;

const CONTEXT_WINDOW_PATTERNS: &[&str] = &[
    "context_length_exceeded",
    "max_tokens",
    "too many tokens",
    "request too large",
    "maximum context length",
    "context window",
    "token limit",
    "content_too_large",
    "request_too_large",
];

const RETRYABLE_SERVER_PATTERNS: &[&str] = &[
    "http 500",
    "http 502",
    "http 503",
    "http 529",
    "server_error",
    "internal server error",
    "overloaded",
    "bad gateway",
    "service unavailable",
];

const RATE_LIMIT_PATTERNS: &[&str] = &[
    "http 429",
    "status=429",
    "status 429",
    "status: 429",
    "too many requests",
    "rate limit",
    "rate_limit",
    "quota exceeded",
];

const SERVER_RETRY_DELAY_MS: u64 = 2_000;
const RATE_LIMIT_INITIAL_RETRY_MS: u64 = 2_000;
const RATE_LIMIT_MAX_RETRY_MS: u64 = 60_000;
const RATE_LIMIT_MAX_RETRIES: u8 = 10;

fn is_context_window_error(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    CONTEXT_WINDOW_PATTERNS.iter().any(|p| lower.contains(p))
}

fn is_retryable_server_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    RETRYABLE_SERVER_PATTERNS.iter().any(|p| lower.contains(p))
}

fn is_rate_limit_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    RATE_LIMIT_PATTERNS.iter().any(|p| lower.contains(p))
}

fn next_rate_limit_backoff(previous_ms: Option<u64>) -> u64 {
    previous_ms
        .map(|ms| ms.saturating_mul(2))
        .unwrap_or(RATE_LIMIT_INITIAL_RETRY_MS)
        .clamp(RATE_LIMIT_INITIAL_RETRY_MS, RATE_LIMIT_MAX_RETRY_MS)
}

fn next_retry_delay_ms(
    msg: &str,
    server_retries: &mut u8,
    rate_limit_retries: &mut u8,
    rate_limit_backoff: &mut Option<u64>,
) -> Option<u64> {
    if is_rate_limit_error(msg) {
        if *rate_limit_retries == 0 {
            return None;
        }
        *rate_limit_retries -= 1;
        let backoff = next_rate_limit_backoff(*rate_limit_backoff);
        *rate_limit_backoff = Some(backoff);
        return Some(backoff);
    }
    if is_retryable_server_error(msg) {
        if *server_retries == 0 {
            return None;
        }
        *server_retries -= 1;
        return Some(SERVER_RETRY_DELAY_MS);
    }
    None
}

// ── Tool result sanitization ────────────────────────────────────────────

const BASE64_TAG: &str = "data:";
const BASE64_MARKER: &str = ";base64,";
const BLOB_MIN_LEN: usize = 200;

fn is_base64_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'='
}

fn strip_base64_blobs(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(start) = rest.find(BASE64_TAG) {
        result.push_str(&rest[..start]);
        let after_tag = &rest[start + BASE64_TAG.len()..];

        if let Some(marker_pos) = after_tag.find(BASE64_MARKER) {
            let mime_part = &after_tag[..marker_pos];
            let payload_start = marker_pos + BASE64_MARKER.len();
            let payload = &after_tag[payload_start..];
            let payload_len = payload.bytes().take_while(|b| is_base64_byte(*b)).count();

            if payload_len >= BLOB_MIN_LEN {
                let total_uri_len = BASE64_TAG.len() + payload_start + payload_len;
                let _ = write!(result, "[{mime_part} data removed — {total_uri_len} bytes]");
                rest = &rest[start + total_uri_len..];
                continue;
            }
        }

        result.push_str(BASE64_TAG);
        rest = after_tag;
    }
    result.push_str(rest);
    result
}

fn strip_hex_blobs(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.char_indices().peekable();

    while let Some(&(start, ch)) = chars.peek() {
        if ch.is_ascii_hexdigit() {
            let mut end = start;
            while let Some(&(i, c)) = chars.peek() {
                if c.is_ascii_hexdigit() {
                    end = i + c.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let run = end - start;
            if run >= BLOB_MIN_LEN {
                let _ = write!(result, "[hex data removed — {run} chars]");
            } else {
                result.push_str(&input[start..end]);
            }
        } else {
            result.push(ch);
            chars.next();
        }
    }
    result
}

fn sanitize_tool_result(input: &str, max_bytes: usize) -> String {
    let mut result = strip_base64_blobs(input);
    result = strip_hex_blobs(&result);

    if result.len() <= max_bytes {
        return result;
    }

    let original_len = result.len();
    let mut end = max_bytes;
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    result.truncate(end);
    let _ = write!(result, "\n\n[truncated — {original_len} bytes total]");
    result
}

// ── Public types ────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum AgentRunError {
    #[error("context window exceeded: {0}")]
    ContextWindowExceeded(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug)]
pub struct AgentRunResult {
    pub text: String,
    pub iterations: usize,
    pub tool_calls_made: usize,
    pub usage: Usage,
    pub available_tools: Vec<ToolDef>,
}

impl AgentRunResult {
    pub fn to_agent_response(&self, session: String) -> cyber_agent_proto::AgentResponse {
        cyber_agent_proto::AgentResponse {
            session,
            text: self.text.clone(),
            iterations: self.iterations as u32,
            tool_calls_made: self.tool_calls_made as u32,
            error: String::new(),
            available_tools: self.available_tools.clone(),
        }
    }
}

impl AgentRunError {
    pub fn to_agent_response(&self, session: String) -> cyber_agent_proto::AgentResponse {
        cyber_agent_proto::AgentResponse {
            session,
            text: String::new(),
            iterations: 0,
            tool_calls_made: 0,
            error: self.to_string(),
            available_tools: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub enum RunnerEvent {
    Iteration(usize),
    ToolCallStart {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolCallEnd {
        id: String,
        name: String,
        success: bool,
        error: Option<String>,
    },
}

pub type OnEvent = Box<dyn Fn(RunnerEvent) + Send + Sync>;

// ── Agent loop ──────────────────────────────────────────────────────────

pub async fn run_agent_loop(
    provider: Arc<dyn LlmProvider>,
    tools: &ToolRegistry,
    system_prompt: &str,
    user_text: &str,
    on_event: Option<&OnEvent>,
    history: Option<Vec<ChatMessage>>,
) -> Result<AgentRunResult, AgentRunError> {
    let tool_defs = tools.list_tool_defs();
    let tools_for_api: &[ToolDef] = if provider.supports_tools() {
        &tool_defs
    } else {
        &[]
    };

    let mut messages: Vec<ChatMessage> = vec![ChatMessage::system(system_prompt)];
    if let Some(hist) = history {
        messages.extend(hist);
    }
    messages.push(ChatMessage::user(user_text));

    let mut iterations = 0usize;
    let mut total_tool_calls = 0usize;
    let mut total_input_tokens: u32 = 0;
    let mut total_output_tokens: u32 = 0;
    let mut server_retries: u8 = 1;
    let mut rate_limit_retries: u8 = RATE_LIMIT_MAX_RETRIES;
    let mut rate_limit_backoff: Option<u64> = None;

    loop {
        iterations += 1;
        if iterations > DEFAULT_MAX_ITERATIONS {
            return Err(AgentRunError::Other(anyhow!(
                "agent loop exceeded max iterations"
            )));
        }

        if let Some(cb) = on_event {
            cb(RunnerEvent::Iteration(iterations));
        }

        let response: CompletionResponse = match provider.complete(&messages, tools_for_api).await
        {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                if is_context_window_error(&msg) {
                    return Err(AgentRunError::ContextWindowExceeded(msg));
                }
                if let Some(delay_ms) = next_retry_delay_ms(
                    &msg,
                    &mut server_retries,
                    &mut rate_limit_retries,
                    &mut rate_limit_backoff,
                ) {
                    iterations -= 1;
                    futures_timer::Delay::new(std::time::Duration::from_millis(delay_ms)).await;
                    continue;
                }
                return Err(AgentRunError::Other(e));
            }
        };

        let usage = response.usage.unwrap_or_default();
        total_input_tokens = total_input_tokens.saturating_add(usage.input_tokens);
        total_output_tokens = total_output_tokens.saturating_add(usage.output_tokens);

        if response.tool_calls.is_empty() {
            return Ok(AgentRunResult {
                text: response.text.clone(),
                iterations,
                tool_calls_made: total_tool_calls,
                usage: Usage {
                    input_tokens: total_input_tokens,
                    output_tokens: total_output_tokens,
                },
                available_tools: tool_defs.clone(),
            });
        }

        messages.push(ChatMessage::assistant_with_tools(
            if response.text.is_empty() { None } else { Some(response.text.clone()) },
            response.tool_calls.clone(),
        ));

        total_tool_calls += response.tool_calls.len();

        for tc in &response.tool_calls {
            if let Some(cb) = on_event {
                cb(RunnerEvent::ToolCallStart {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments(),
                });
            }
        }

        let call_futures: Vec<_> = response
            .tool_calls
            .iter()
            .map(|tc| tools.execute_call(&tc.id, &tc.name, &tc.arguments_json))
            .collect();

        let results = futures::future::join_all(call_futures).await;

        for result in &results {
            if let Some(cb) = on_event {
                let error = if result.success { None } else { Some(result.error.clone()) };
                cb(RunnerEvent::ToolCallEnd {
                    id: result.id.clone(),
                    name: response.tool_calls.iter().find(|tc| tc.id == result.id).map(|tc| tc.name.clone()).unwrap_or_default(),
                    success: result.success,
                    error,
                });
            }

            let content = if result.success { &result.result_json } else { &result.error };
            let tool_result_str = sanitize_tool_result(content, MAX_TOOL_RESULT_BYTES);
            messages.push(ChatMessage::tool(&result.id, &tool_result_str));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cyber_agent_proto::{ToolCall, ToolDef};
    use cyber_agent_tool::AgentTool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockProvider {
        response_text: String,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }
        fn id(&self) -> &str {
            "mock-model"
        }
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ToolDef],
        ) -> anyhow::Result<CompletionResponse> {
            Ok(CompletionResponse {
                text: self.response_text.clone(),
                tool_calls: vec![],
                usage: Some(Usage { input_tokens: 10, output_tokens: 5 }),
            })
        }
    }

    struct ToolCallingProvider {
        call_count: AtomicUsize,
    }

    #[async_trait]
    impl LlmProvider for ToolCallingProvider {
        fn name(&self) -> &str { "mock" }
        fn id(&self) -> &str { "mock-model" }
        fn supports_tools(&self) -> bool { true }
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ToolDef],
        ) -> anyhow::Result<CompletionResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(CompletionResponse {
                    text: String::new(),
                    tool_calls: vec![ToolCall::new("call_1", "echo_tool", serde_json::json!({"text": "hi"}))],
                    usage: Some(Usage { input_tokens: 10, output_tokens: 5 }),
                })
            } else {
                Ok(CompletionResponse {
                    text: "Done!".into(),
                    tool_calls: vec![],
                    usage: Some(Usage { input_tokens: 20, output_tokens: 10 }),
                })
            }
        }
    }

    struct EchoTool;

    #[async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str { "echo_tool" }
        fn description(&self) -> &str { "Echoes input" }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        async fn execute(&self, params: serde_json::Value) -> anyhow::Result<serde_json::Value> {
            Ok(params)
        }
    }

    #[tokio::test]
    async fn text_response_no_tools() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider {
            response_text: "Hello!".into(),
        });
        let tools = ToolRegistry::new();

        let result = run_agent_loop(provider, &tools, "system", "hi", None, None)
            .await
            .unwrap();

        assert_eq!(result.text, "Hello!");
        assert_eq!(result.iterations, 1);
        assert_eq!(result.tool_calls_made, 0);
    }

    #[tokio::test]
    async fn tool_call_loop() {
        let provider: Arc<dyn LlmProvider> = Arc::new(ToolCallingProvider {
            call_count: AtomicUsize::new(0),
        });
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(EchoTool));

        let result = run_agent_loop(provider, &tools, "system", "test", None, None)
            .await
            .unwrap();

        assert_eq!(result.text, "Done!");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tool_calls_made, 1);
        assert_eq!(result.usage.input_tokens, 30);
        assert_eq!(result.usage.output_tokens, 15);
    }

    #[test]
    fn sanitize_truncates_long_result() {
        let input = "hello world! ".repeat(20_000);
        let result = sanitize_tool_result(&input, 1000);
        assert!(result.len() < 1100);
        assert!(result.contains("[truncated"));
    }

    #[test]
    fn sanitize_strips_base64_blobs() {
        let base64_data = "A".repeat(300);
        let input = format!("before data:image/png;base64,{} after", base64_data);
        let result = sanitize_tool_result(&input, 100_000);
        assert!(result.contains("removed"));
        assert!(!result.contains(&base64_data));
    }

    #[test]
    fn sanitize_strips_hex_blobs() {
        let hex_data = "a1b2c3d4".repeat(50);
        let input = format!("prefix {} suffix", hex_data);
        let result = sanitize_tool_result(&input, 100_000);
        assert!(result.contains("[hex data removed"));
        assert!(!result.contains(&hex_data));
    }
}
