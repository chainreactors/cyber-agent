use std::sync::Arc;

use anyhow::{anyhow, bail};
use async_trait::async_trait;

use cyber_agent_proto::{
    ChatMessage, CompletionResponse, LlmProvider, ToolCall, ToolDef, Usage,
};
use cyber_agent_protocol::{
    BridgeCompletionPayload, BridgeRequest, BridgeRequestFunction, BridgeRequestTool,
    BridgeResponse, BridgeResponsePayload,
};
use cyber_agent_transport::Transport;

#[derive(Clone)]
pub struct BridgeProvider {
    #[allow(dead_code)]
    endpoint: String,
    model_id: String,
    provider_name: String,
    transport: Arc<dyn Transport>,
}

impl BridgeProvider {
    pub fn new(
        endpoint: String,
        model_id: String,
        provider_name: String,
        transport: Arc<dyn Transport>,
    ) -> Self {
        Self {
            endpoint,
            model_id,
            provider_name,
            transport,
        }
    }

    fn build_request(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
    ) -> BridgeRequest {
        BridgeRequest {
            model: self.model_id.clone(),
            messages: messages.iter().map(|m| serde_json::to_value(m).unwrap_or_default()).collect(),
            tools: tools.iter().map(tool_def_to_request_tool).collect(),
        }
    }
}

fn tool_def_to_request_tool(def: &ToolDef) -> BridgeRequestTool {
    let mut properties = serde_json::Map::new();
    let mut required = vec![];
    for p in &def.params {
        properties.insert(
            p.name.clone(),
            serde_json::json!({
                "type": &p.r#type,
                "description": &p.description,
            }),
        );
        if p.required {
            required.push(serde_json::Value::String(p.name.clone()));
        }
    }
    BridgeRequestTool {
        tool_type: "function".to_string(),
        function: BridgeRequestFunction {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required,
            }),
            strict: false,
        },
    }
}

fn parse_tool_calls(
    tool_calls: &[cyber_agent_protocol::BridgeToolCall],
) -> Vec<ToolCall> {
    tool_calls
        .iter()
        .filter_map(|tc| {
            if tc.function.name.is_empty() {
                return None;
            }
            Some(ToolCall {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments_json: tc.function.arguments.clone(),
            })
        })
        .collect()
}

fn completion_from_payload(payload: BridgeCompletionPayload) -> anyhow::Result<CompletionResponse> {
    let message = payload
        .choices
        .first()
        .map(|choice| &choice.message)
        .ok_or_else(|| anyhow!("bridge response missing choices[0].message"))?;

    Ok(CompletionResponse {
        text: message.content.clone().unwrap_or_default(),
        tool_calls: parse_tool_calls(&message.tool_calls),
        usage: Some(Usage {
            input_tokens: payload.usage.prompt_tokens,
            output_tokens: payload.usage.completion_tokens,
        }),
    })
}

#[async_trait]
impl LlmProvider for BridgeProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn id(&self) -> &str {
        &self.model_id
    }

    fn supports_tools(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
    ) -> anyhow::Result<CompletionResponse> {
        let request = self.build_request(messages, tools);
        let request_bytes =
            serde_json::to_vec(&request).map_err(|e| anyhow!("serialize request: {}", e))?;

        let response_bytes = self.transport.request(&request_bytes).await?;

        let response: BridgeResponse = serde_json::from_slice(&response_bytes)
            .map_err(|e| anyhow!("deserialize response: {}", e))?;

        match response.payload {
            BridgeResponsePayload::Error(err) => {
                let message = err
                    .error
                    .message()
                    .unwrap_or_else(|| "unknown bridge error".to_string());
                bail!("bridge provider error: {message}");
            }
            BridgeResponsePayload::Completion(payload) => completion_from_payload(payload),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cyber_agent_protocol::{
        BridgeAssistantMessage, BridgeChoice, BridgeCompletionPayload, BridgeResponse,
        BridgeResponsePayload, BridgeToolCall, BridgeToolCallFunction, BridgeUsage,
    };
    use std::sync::Mutex;

    struct StaticTransport {
        response: BridgeResponse,
        seen: Mutex<Vec<Vec<u8>>>,
    }

    #[async_trait]
    impl Transport for StaticTransport {
        async fn request(&self, data: &[u8]) -> anyhow::Result<Vec<u8>> {
            self.seen.lock().unwrap().push(data.to_vec());
            Ok(serde_json::to_vec(&self.response)?)
        }
    }

    fn text_response(content: &str) -> BridgeResponse {
        BridgeResponse {
            payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                choices: vec![BridgeChoice {
                    message: BridgeAssistantMessage {
                        content: Some(content.to_string()),
                        tool_calls: vec![],
                    },
                    finish_reason: None,
                }],
                usage: BridgeUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                },
            }),
        }
    }

    fn tool_call_response(content: &str, tool_calls: Vec<BridgeToolCall>) -> BridgeResponse {
        BridgeResponse {
            payload: BridgeResponsePayload::Completion(BridgeCompletionPayload {
                choices: vec![BridgeChoice {
                    message: BridgeAssistantMessage {
                        content: Some(content.to_string()),
                        tool_calls,
                    },
                    finish_reason: None,
                }],
                usage: BridgeUsage {
                    prompt_tokens: 12,
                    completion_tokens: 8,
                },
            }),
        }
    }

    #[tokio::test]
    async fn complete_text_response() {
        let transport = Arc::new(StaticTransport {
            response: text_response("hello"),
            seen: Mutex::new(vec![]),
        });
        let provider = BridgeProvider::new(
            "test://".into(),
            "gpt-4o".into(),
            "test".into(),
            transport.clone(),
        );

        let messages = vec![ChatMessage::user("hi")];
        let result = provider.complete(&messages, &[]).await.unwrap();

        assert_eq!(result.text, "hello");
        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert!(result.tool_calls.is_empty());

        let seen = transport.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let sent: BridgeRequest = serde_json::from_slice(&seen[0]).unwrap();
        assert_eq!(sent.model, "gpt-4o");
    }

    #[tokio::test]
    async fn complete_with_tool_calls() {
        let transport = Arc::new(StaticTransport {
            response: tool_call_response(
                "thinking",
                vec![BridgeToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: BridgeToolCallFunction {
                        name: "shell".into(),
                        arguments: r#"{"command":"ls"}"#.into(),
                    },
                }],
            ),
            seen: Mutex::new(vec![]),
        });
        let provider = BridgeProvider::new(
            "test://".into(),
            "gpt-4o".into(),
            "test".into(),
            transport,
        );

        let tools = vec![cyber_agent_proto::ToolDef {
            name: "shell".into(),
            description: "Execute command".into(),
            params: vec![cyber_agent_proto::ToolParam {
                name: "command".into(),
                description: "Command to run".into(),
                r#type: "string".into(),
                required: true,
            }],
        }];
        let result = provider
            .complete(&[ChatMessage::user("run ls")], &tools)
            .await
            .unwrap();

        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "shell");
        assert_eq!(result.tool_calls[0].arguments()["command"], "ls");
    }
}
