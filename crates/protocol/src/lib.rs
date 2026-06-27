use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<BridgeRequestTool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRequestTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: BridgeRequestFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRequestFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeResponse {
    pub payload: BridgeResponsePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BridgeResponsePayload {
    Error(BridgeErrorPayload),
    Completion(BridgeCompletionPayload),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeErrorPayload {
    pub error: BridgeErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BridgeErrorBody {
    Message { message: String },
    Text(String),
    Json(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BridgeCompletionPayload {
    #[serde(default)]
    pub choices: Vec<BridgeChoice>,
    #[serde(default)]
    pub usage: BridgeUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeChoice {
    pub message: BridgeAssistantMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BridgeAssistantMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<BridgeToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: BridgeToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BridgeUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

impl BridgeErrorBody {
    pub fn message(&self) -> Option<String> {
        match self {
            BridgeErrorBody::Message { message } => {
                (!message.is_empty()).then(|| message.clone())
            }
            BridgeErrorBody::Text(message) => (!message.is_empty()).then(|| message.clone()),
            BridgeErrorBody::Json(value) => {
                if let Some(message) = value
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .filter(|m| !m.is_empty())
                {
                    return Some(message.to_string());
                }
                if value.is_object() || value.is_array() {
                    return Some(value.to_string());
                }
                None
            }
        }
    }
}
