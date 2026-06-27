pub mod agentpb {
    include!(concat!(env!("OUT_DIR"), "/agentpb.rs"));
}

pub use agentpb::*;

#[cfg(feature = "grpc")]
pub use agentpb::agent_service_client::AgentServiceClient;
#[cfg(feature = "grpc")]
pub use agentpb::agent_service_server::{AgentService, AgentServiceServer};

mod message;
mod provider;
mod transport;
pub use provider::LlmProvider;
pub use transport::Transport;
