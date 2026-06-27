pub mod agentpb {
    include!(concat!(env!("OUT_DIR"), "/agentpb.rs"));
}

pub use agentpb::*;
