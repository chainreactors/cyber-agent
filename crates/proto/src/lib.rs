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
pub use provider::LlmProvider;
