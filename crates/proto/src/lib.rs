pub mod agentpb {
    include!(concat!(env!("OUT_DIR"), "/agentpb.rs"));
}

pub use agentpb::*;

mod message;
mod provider;
pub use provider::LlmProvider;
