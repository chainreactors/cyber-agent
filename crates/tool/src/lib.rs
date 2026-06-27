use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use cyber_agent_proto::{ToolDef, ToolParam};

#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn AgentTool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn AgentTool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::from(tool));
    }

    pub fn get(&self, name: &str) -> Option<&dyn AgentTool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn list_schemas(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters_schema(),
                })
            })
            .collect()
    }

    pub fn list_tool_defs(&self) -> Vec<ToolDef> {
        self.tools
            .values()
            .map(|tool| schema_to_tool_def(tool.as_ref()))
            .collect()
    }

    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

fn schema_to_tool_def(tool: &dyn AgentTool) -> ToolDef {
    let schema = tool.parameters_schema();
    let params = schema["properties"]
        .as_object()
        .map(|props| {
            let required: Vec<&str> = schema["required"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            props
                .iter()
                .map(|(k, v)| ToolParam {
                    name: k.clone(),
                    description: v["description"].as_str().unwrap_or("").into(),
                    r#type: v["type"].as_str().unwrap_or("string").into(),
                    required: required.contains(&k.as_str()),
                })
                .collect()
        })
        .unwrap_or_default();

    ToolDef {
        name: tool.name().into(),
        description: tool.description().into(),
        params,
    }
}

/// Create an `AgentTool` from a proto `ToolDef`.
///
/// Server-defined tools whose schema is forwarded to the LLM.
/// Execution returns an error since they run on the server side.
pub struct DynamicTool {
    name: String,
    description: String,
    schema: serde_json::Value,
}

impl DynamicTool {
    pub fn from_tool_def(def: &ToolDef) -> Self {
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
                required.push(serde_json::json!(&p.name));
            }
        }
        Self {
            name: def.name.clone(),
            description: def.description.clone(),
            schema: serde_json::json!({"type": "object", "properties": properties, "required": required}),
        }
    }
}

#[async_trait]
impl AgentTool for DynamicTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn execute(&self, _params: serde_json::Value) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"error": format!("tool '{}' is server-defined, not executable locally", self.name)}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool {
        tool_name: String,
    }

    #[async_trait]
    impl AgentTool for DummyTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "first arg"}
                },
                "required": ["arg1"]
            })
        }

        async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
            Ok(params)
        }
    }

    #[test]
    fn register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool {
            tool_name: "test".into(),
        }));
        assert!(registry.get("test").is_some());
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn list_schemas_and_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool {
            tool_name: "alpha".into(),
        }));
        registry.register(Box::new(DummyTool {
            tool_name: "beta".into(),
        }));

        let schemas = registry.list_schemas();
        assert_eq!(schemas.len(), 2);

        let mut names = registry.list_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn list_tool_defs_matches_proto() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool {
            tool_name: "shell".into(),
        }));

        let defs = registry.list_tool_defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "shell");
        assert_eq!(defs[0].description, "test tool");
        assert_eq!(defs[0].params.len(), 1);
        assert_eq!(defs[0].params[0].name, "arg1");
        assert_eq!(defs[0].params[0].r#type, "string");
        assert!(defs[0].params[0].required);
    }

    #[test]
    fn dynamic_tool_round_trip() {
        let def = ToolDef {
            name: "remote_scan".into(),
            description: "Scan a target".into(),
            params: vec![
                ToolParam {
                    name: "target".into(),
                    description: "IP or hostname".into(),
                    r#type: "string".into(),
                    required: true,
                },
                ToolParam {
                    name: "port".into(),
                    description: "Port number".into(),
                    r#type: "integer".into(),
                    required: false,
                },
            ],
        };

        let tool = DynamicTool::from_tool_def(&def);
        assert_eq!(tool.name(), "remote_scan");
        assert_eq!(tool.description(), "Scan a target");

        let schema = tool.parameters_schema();
        assert_eq!(schema["properties"]["target"]["type"], "string");
        assert_eq!(schema["properties"]["port"]["type"], "integer");

        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "target"));
        assert!(!required.iter().any(|v| v == "port"));

        // Round-trip back to ToolDef
        let defs = {
            let mut reg = ToolRegistry::new();
            reg.register(Box::new(tool));
            reg.list_tool_defs()
        };
        assert_eq!(defs[0].name, "remote_scan");
        assert_eq!(defs[0].params.len(), 2);
    }
}
