const PROTO_FILE: &str = "../../proto/agent/agentpb/agent.proto";
const PROTO_INCLUDE: &str = "../../proto/";

fn main() {
    let mut config = prost_build::Config::new();
    config.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]");
    config.field_attribute(".agentpb", "#[serde(default)]");

    config
        .compile_protos(&[PROTO_FILE], &[PROTO_INCLUDE])
        .expect("failed to compile agent.proto");

    fix_serde_enum_variants();
}

fn fix_serde_enum_variants() {
    use std::fs;
    let out_dir = std::env::var("OUT_DIR").unwrap();
    for entry in fs::read_dir(&out_dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |ext| ext != "rs") {
            continue;
        }
        let content = fs::read_to_string(&path).unwrap();
        let mut lines: Vec<&str> = content.lines().collect();
        let mut to_remove = Vec::new();
        let mut in_enum = false;
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("pub enum ") {
                in_enum = true;
            } else if in_enum && trimmed == "}" {
                in_enum = false;
            } else if in_enum && trimmed == "#[serde(default)]" {
                to_remove.push(i);
            }
        }
        if !to_remove.is_empty() {
            for &i in to_remove.iter().rev() {
                lines.remove(i);
            }
            fs::write(&path, lines.join("\n") + "\n").unwrap();
        }
    }
}
