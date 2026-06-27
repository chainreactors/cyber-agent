use async_trait::async_trait;

#[derive(Debug, Clone)]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: UserContent,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone)]
pub enum UserContent {
    Text(String),
    Multimodal(Vec<ContentPart>),
}

impl UserContent {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }
}

#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    Image { media_type: String, data: String },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: UserContent::Text(content.into()),
        }
    }

    pub fn user_multimodal(parts: Vec<ContentPart>) -> Self {
        Self::User {
            content: UserContent::Multimodal(parts),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls: vec![],
        }
    }

    pub fn assistant_with_tools(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content,
            tool_calls,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    pub fn to_openai_value(&self) -> serde_json::Value {
        match self {
            ChatMessage::System { content } => {
                serde_json::json!({ "role": "system", "content": content })
            }
            ChatMessage::User { content } => match content {
                UserContent::Text(text) => {
                    serde_json::json!({ "role": "user", "content": text })
                }
                UserContent::Multimodal(parts) => {
                    let blocks: Vec<serde_json::Value> = parts
                        .iter()
                        .map(|part| match part {
                            ContentPart::Text(text) => {
                                serde_json::json!({ "type": "text", "text": text })
                            }
                            ContentPart::Image { media_type, data } => {
                                let data_uri = format!("data:{media_type};base64,{data}");
                                serde_json::json!({
                                    "type": "image_url",
                                    "image_url": { "url": data_uri }
                                })
                            }
                        })
                        .collect();
                    serde_json::json!({ "role": "user", "content": blocks })
                }
            },
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => {
                if tool_calls.is_empty() {
                    serde_json::json!({
                        "role": "assistant",
                        "content": content.as_deref().unwrap_or(""),
                    })
                } else {
                    let tc_json: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    let mut msg = serde_json::json!({
                        "role": "assistant",
                        "tool_calls": tc_json,
                    });
                    if let Some(text) = content {
                        msg["content"] = serde_json::Value::String(text.clone());
                    }
                    msg
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug)]
pub struct CompletionResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn id(&self) -> &str;

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> anyhow::Result<CompletionResponse>;

    fn supports_tools(&self) -> bool {
        false
    }
}
