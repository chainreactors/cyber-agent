use async_trait::async_trait;

use crate::{ChatMessage, CompletionResponse, ToolDef};

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn id(&self) -> &str;

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
    ) -> anyhow::Result<CompletionResponse>;

    fn supports_tools(&self) -> bool {
        false
    }
}
