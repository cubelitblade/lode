use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output};
use std::thread;

use serde_json::Value;
use tempfile::TempDir;

const SOURCE: &str = "# Heading\n\nraw **Markdown** body";

fn run_lode(workspace: &Path, config_home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_lode"))
        .arg("--workspace")
        .arg(workspace)
        .arg("--view")
        .arg("json")
        .args(args)
        .env("XDG_CONFIG_HOME", config_home)
        .output()
        .unwrap()
}

fn mock_embedding_server() -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let handle = thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let mut stream = stream.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let count = stream.read(&mut buffer).unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request);
            assert!(request.starts_with("POST /v1/embeddings"));
            let body = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[1.0,0.0,0.0,0.0]}],"model":"mock"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
    });
    (address, handle)
}

#[test]
fn markdown_mine_prospect_dig_process_flow_preserves_raw_text() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    std::fs::create_dir_all(config_home.path().join("lode")).unwrap();
    std::fs::write(
        config_home.path().join("lode/config.toml"),
        "[embedding]\nprovider = \"openai_compatible\"\nmodel = \"mock-model\"\nmodel_dimension = 4\n[embedding.openai_compatible]\nendpoint = \"MOCK_ENDPOINT\"\nmax_retries = 0\n[fts]\nstrategy = \"unicode61\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path().join("guide.md"), SOURCE).unwrap();
    let (endpoint, server) = mock_embedding_server();
    let config_path = config_home.path().join("lode/config.toml");
    let config = std::fs::read_to_string(&config_path)
        .unwrap()
        .replace("MOCK_ENDPOINT", &endpoint);
    std::fs::write(config_path, config).unwrap();

    let mine = run_lode(workspace.path(), config_home.path(), &["mine"]);
    assert!(
        mine.status.success(),
        "mine failed: {}",
        String::from_utf8_lossy(&mine.stderr)
    );
    let mine_json: Value = serde_json::from_slice(&mine.stdout).unwrap();
    assert_eq!(mine_json["failed"], serde_json::json!([]));
    assert_eq!(mine_json["added"][0], "guide.md");

    let prospect = run_lode(workspace.path(), config_home.path(), &["prospect", "raw"]);
    assert!(
        prospect.status.success(),
        "prospect failed: {}",
        String::from_utf8_lossy(&prospect.stderr)
    );
    let prospect_json: Value = serde_json::from_slice(&prospect.stdout).unwrap();
    let hit = &prospect_json["hits"][0];
    assert_eq!(hit["paths"][0]["path"], "guide.md");
    assert_eq!(hit["heading"], "Heading");
    assert!(hit["preview"].as_str().unwrap().contains("# Heading"));
    assert!(hit["preview"].as_str().unwrap().contains("**Markdown**"));
    let digest = hit["digest"].as_str().unwrap();

    let dig = run_lode(workspace.path(), config_home.path(), &["dig", digest]);
    assert!(
        dig.status.success(),
        "dig failed: {}",
        String::from_utf8_lossy(&dig.stderr)
    );
    let dig_json: Value = serde_json::from_slice(&dig.stdout).unwrap();
    let chunk = &dig_json["window"]["chunks"][0];
    assert_eq!(chunk["paths"][0]["path"], "guide.md");
    assert_eq!(chunk["heading"], "Heading");
    assert_eq!(chunk["text"], SOURCE);

    server.join().unwrap();
}
