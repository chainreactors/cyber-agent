use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Transport: Send + Sync {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>>;
}
