use anyhow::Result;
use async_trait::async_trait;

/// Unified transport trait supporting both request-response and bidirectional modes.
///
/// Implementors choose which methods to support based on transport capabilities:
/// - TCP/HTTP: implement `request()` (default `send`/`recv` split it automatically)
/// - WebSocket/gRPC/C2 channel: implement `send()` + `recv()` (default `request` combines them)
///
/// At least one pair must be implemented. The defaults call each other,
/// so implementing just one side works for simple transports.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Request-response: send data, wait for response. Default: send + recv.
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.send(data).await?;
        self.recv().await
    }

    /// Send data without waiting for a response. Default: calls request and drops the response.
    async fn send(&self, data: &[u8]) -> Result<()> {
        let _ = self.request(data).await?;
        Ok(())
    }

    /// Receive data. Default: calls request with empty payload.
    async fn recv(&self) -> Result<Vec<u8>> {
        self.request(&[]).await
    }
}
