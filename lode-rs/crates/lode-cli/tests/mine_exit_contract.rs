use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

use lode_core::index::records::{FileRecord, FileStatus};
use lode_core::index::store::Store;
use lode_core::ingestion::digest::file_digest;
use lode_core::ingestion::types::Chunk;
use lode_core::relpath::WorkspacePath;
use serde_json::Value;
use tempfile::TempDir;

const DIMENSION: usize = 4;
const MODEL: &str = "contract-test-model";
const VIEWS: [&str; 4] = ["compact", "extended", "table", "json"];
const OLD_CHUNK_DIGEST: &str =
    "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OLD_CHUNK_TEXT: &str = "old docx snapshot";

struct MockEmbeddingServer {
    endpoint: String,
    thread: thread::JoinHandle<()>,
}

impl MockEmbeddingServer {
    fn join(self) {
        self.thread
            .join()
            .expect("embedding server thread panicked");
    }
}

fn run_lode(workspace: &Path, config_home: &Path, view: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_lode"))
        .arg("--workspace")
        .arg(workspace)
        .arg("--view")
        .arg(view)
        .args(args)
        .env("XDG_CONFIG_HOME", config_home)
        .output()
        .expect("failed to start lode")
}

fn write_config(workspace: &Path, endpoint: &str) {
    fs::write(
        workspace.join("lode.toml"),
        format!(
            "[embedding]\nprovider = \"openai_compatible\"\nmodel = \"{MODEL}\"\nmodel_dimension = {DIMENSION}\nbatch_size = 32\n[embedding.openai_compatible]\nendpoint = \"{endpoint}\"\nmax_retries = 0\ntimeout = 2.0\n[fts]\nstrategy = \"unicode61\"\n"
        ),
    )
    .expect("write lode.toml");
}

fn spawn_embedding_server(expected_requests: usize) -> MockEmbeddingServer {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind local mock server");
    listener
        .set_nonblocking(true)
        .expect("make mock listener nonblocking");
    let endpoint = format!(
        "http://127.0.0.1:{}",
        listener.local_addr().expect("mock listener address").port()
    );
    let thread = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut handled = 0;
        while handled < expected_requests && Instant::now() < deadline {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .expect("set mock read timeout");
                    stream
                        .set_write_timeout(Some(Duration::from_secs(2)))
                        .expect("set mock write timeout");
                    let (path, body) = read_request(&mut stream);
                    assert_eq!(path, "/v1/embeddings");
                    let request: Value =
                        serde_json::from_slice(&body).expect("embedding request JSON");
                    let inputs = request["input"]
                        .as_array()
                        .expect("embedding request input array");
                    let data: Vec<Value> = inputs
                        .iter()
                        .enumerate()
                        .map(|(index, _)| {
                            serde_json::json!({
                                "object": "embedding",
                                "index": index,
                                "embedding": [1.0, 0.0, 0.0, 0.0],
                            })
                        })
                        .collect();
                    let response_body = serde_json::json!({ "data": data }).to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream
                        .write_all(response.as_bytes())
                        .expect("write mock response");
                    handled += 1;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept mock request: {error}"),
            }
        }
        assert_eq!(
            handled, expected_requests,
            "mock server timed out waiting for requests"
        );
    });
    MockEmbeddingServer { endpoint, thread }
}

fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let header_end = loop {
        let count = stream.read(&mut buffer).expect("read mock request");
        assert!(count > 0, "mock client closed before sending headers");
        request.extend_from_slice(&buffer[..count]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };

    let header = String::from_utf8_lossy(&request[..header_end]).into_owned();
    let content_length = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("valid content length"))
        })
        .expect("content-length header");
    while request.len() < header_end + content_length {
        let count = stream.read(&mut buffer).expect("read mock request body");
        assert!(count > 0, "mock client closed before sending body");
        request.extend_from_slice(&buffer[..count]);
    }
    let body = request[header_end..header_end + content_length].to_vec();
    let path = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("request target")
        .to_string();
    (path, body)
}

fn json_output(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON stdout ({error}); stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn assert_mine_fields(payload: &Value) {
    let object = payload.as_object().expect("mine JSON object");
    assert_eq!(object.len(), 5, "unexpected mine JSON fields: {object:?}");
    for field in ["added", "updated", "removed", "renamed", "failed"] {
        assert!(
            object.contains_key(field),
            "missing mine JSON field {field}"
        );
        assert!(
            object[field].is_array(),
            "mine JSON field {field} is not an array"
        );
    }
}

fn assert_success_view(view: &str, output: &Output, path: &str) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "{view} failed: {}",
        stderr(output)
    );
    match view {
        "compact" | "extended" => {
            let stdout = stdout(output);
            assert!(stdout.contains("Mining complete."), "{view}: {stdout}");
            assert!(stdout.contains(path), "{view}: {stdout}");
        }
        "table" => {
            let stdout = stdout(output);
            assert!(stdout.starts_with("STATUS\tPATH\tFROM\tDETAIL\n"));
            assert!(
                stdout
                    .lines()
                    .any(|line| line.starts_with(&format!("added\t{path}\t")))
            );
        }
        "json" => {
            let payload = json_output(output);
            assert_mine_fields(&payload);
            assert_eq!(payload["added"], serde_json::json!([path]));
            assert_eq!(payload["updated"], serde_json::json!([]));
            assert_eq!(payload["removed"], serde_json::json!([]));
            assert_eq!(payload["renamed"], serde_json::json!([]));
            assert_eq!(payload["failed"], serde_json::json!([]));
        }
        _ => unreachable!("test view list is exhaustive"),
    }
}

fn assert_noop_view(view: &str, output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "{view} no-op failed: {}",
        stderr(output)
    );
    match view {
        "compact" | "extended" => assert!(stdout(output).contains("Nothing to do.")),
        "table" => assert_eq!(stdout(output), "STATUS\tPATH\tFROM\tDETAIL\n"),
        "json" => {
            let payload = json_output(output);
            assert_mine_fields(&payload);
            for field in ["added", "updated", "removed", "renamed", "failed"] {
                assert_eq!(
                    payload[field],
                    serde_json::json!([]),
                    "non-empty no-op {field}"
                );
            }
        }
        _ => unreachable!("test view list is exhaustive"),
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn mine_success_and_noop_use_exit_zero_in_every_view() {
    for view in VIEWS {
        let workspace = TempDir::new().expect("workspace tempdir");
        let config_home = TempDir::new().expect("config tempdir");
        fs::write(workspace.path().join("guide.txt"), "a guide").expect("write source");
        let server = spawn_embedding_server(1);
        write_config(workspace.path(), &server.endpoint);

        let first = run_lode(workspace.path(), config_home.path(), view, &["mine"]);
        assert_success_view(view, &first, "guide.txt");
        server.join();

        let second = run_lode(workspace.path(), config_home.path(), view, &["mine"]);
        assert_noop_view(view, &second);
    }
}

fn old_docx_record(path: &Path, workspace: &Path, digest: String) -> FileRecord {
    let metadata = fs::metadata(path).expect("old docx metadata");
    let mtime = metadata
        .modified()
        .expect("old docx mtime")
        .duration_since(UNIX_EPOCH)
        .expect("old docx mtime before epoch")
        .as_secs_f64();
    FileRecord {
        path: WorkspacePath::from_native(path.strip_prefix(workspace).expect("relative docx")),
        digest,
        mtime,
        size: metadata.len(),
        status: FileStatus::Fresh,
    }
}

#[test]
fn mine_partial_failure_keeps_success_and_stale_old_docx_in_every_view() {
    for view in VIEWS {
        let workspace = TempDir::new().expect("workspace tempdir");
        let config_home = TempDir::new().expect("config tempdir");
        let broken = workspace.path().join("broken.docx");
        fs::write(&broken, b"old docx bytes").expect("write old docx");
        fs::create_dir_all(workspace.path().join(".lode")).expect("create index directory");
        let old_digest = file_digest(b"old docx bytes");
        let record = old_docx_record(&broken, workspace.path(), old_digest);
        let mut store = Store::open(
            &workspace.path().join(".lode/index.db"),
            MODEL,
            DIMENSION as u32,
            "unicode61",
        )
        .expect("create old index");
        let old_chunk = Chunk {
            digest: OLD_CHUNK_DIGEST.to_string(),
            text: OLD_CHUNK_TEXT.to_string(),
            seq: 0,
            heading: String::new(),
            page: None,
            span: None,
        };
        store
            .replace_file(
                &record,
                "docx",
                &[old_chunk],
                Some(&[vec![1.0, 0.0, 0.0, 0.0]]),
            )
            .expect("store old docx snapshot");
        drop(store);

        fs::write(&broken, b"this is definitely not a docx").expect("corrupt docx");
        fs::write(workspace.path().join("good.txt"), "new good content").expect("write good file");
        let server = spawn_embedding_server(1);
        write_config(workspace.path(), &server.endpoint);

        let output = run_lode(workspace.path(), config_home.path(), view, &["mine"]);
        assert_eq!(
            output.status.code(),
            Some(3),
            "{view} partial failed: {}",
            stderr(&output)
        );
        match view {
            "compact" | "extended" => {
                let text = stdout(&output);
                assert!(text.contains("good.txt"), "{view}: {text}");
                assert!(text.contains("broken.docx"), "{view}: {text}");
            }
            "table" => {
                let text = stdout(&output);
                assert!(
                    text.lines()
                        .any(|line| line.starts_with("added\tgood.txt\t"))
                );
                assert!(
                    text.lines()
                        .any(|line| line.starts_with("failed\tbroken.docx\t"))
                );
            }
            "json" => {
                let payload = json_output(&output);
                assert_mine_fields(&payload);
                assert_eq!(payload["added"], serde_json::json!(["good.txt"]));
                assert_eq!(payload["updated"], serde_json::json!([]));
                assert_eq!(payload["removed"], serde_json::json!([]));
                assert_eq!(payload["renamed"], serde_json::json!([]));
                assert_eq!(payload["failed"].as_array().map(Vec::len), Some(1));
                assert_eq!(payload["failed"][0]["path"], "broken.docx");
                assert!(payload["failed"][0]["error"].is_string());
            }
            _ => unreachable!("test view list is exhaustive"),
        }
        server.join();

        let store = Store::open_existing(&workspace.path().join(".lode/index.db"))
            .expect("reopen partial index");
        let files = store.list_files().expect("list partial index files");
        let good = files.iter().find(|file| file.path.as_str() == "good.txt");
        assert_eq!(good.map(|file| file.status), Some(FileStatus::Fresh));
        let broken_record = files
            .iter()
            .find(|file| file.path.as_str() == "broken.docx")
            .expect("failed old docx remains indexed");
        assert_eq!(broken_record.status, FileStatus::Stale);
        let old_chunks = store
            .find_chunks_by_digest(OLD_CHUNK_DIGEST.strip_prefix("blake3:").unwrap())
            .expect("read old stale chunk");
        assert_eq!(old_chunks.len(), 1);
        assert_eq!(old_chunks[0].text, OLD_CHUNK_TEXT);
        assert_eq!(old_chunks[0].refs[0].status, FileStatus::Stale);
    }
}

#[test]
fn mine_all_failure_uses_exit_three_in_every_view() {
    for view in VIEWS {
        let workspace = TempDir::new().expect("workspace tempdir");
        let config_home = TempDir::new().expect("config tempdir");
        fs::write(workspace.path().join("broken.docx"), b"not a docx").expect("write broken docx");
        // Extraction fails before embedding, so no server is needed for this
        // scenario. Keep the endpoint local in case a regression attempts a
        // request; the configured model and dimension avoid metadata probes.
        write_config(workspace.path(), "http://127.0.0.1:1");

        let output = run_lode(workspace.path(), config_home.path(), view, &["mine"]);
        assert_eq!(
            output.status.code(),
            Some(3),
            "{view} all-failure code: {}",
            stderr(&output)
        );
        match view {
            "compact" | "extended" => assert!(stdout(&output).contains("broken.docx")),
            "table" => assert!(stdout(&output).contains("failed\tbroken.docx\t")),
            "json" => {
                let payload = json_output(&output);
                assert_mine_fields(&payload);
                assert_eq!(payload["added"], serde_json::json!([]));
                assert_eq!(payload["updated"], serde_json::json!([]));
                assert_eq!(payload["removed"], serde_json::json!([]));
                assert_eq!(payload["renamed"], serde_json::json!([]));
                assert_eq!(payload["failed"].as_array().map(Vec::len), Some(1));
                assert_eq!(payload["failed"][0]["path"], "broken.docx");
            }
            _ => unreachable!("test view list is exhaustive"),
        }
    }
}

#[test]
fn mine_bad_configuration_is_fatal_exit_one_in_every_view() {
    for view in VIEWS {
        let workspace = TempDir::new().expect("workspace tempdir");
        let config_home = TempDir::new().expect("config tempdir");
        fs::write(workspace.path().join("guide.txt"), "guide").expect("write source");
        fs::write(workspace.path().join("lode.toml"), "[embedding\n").expect("write bad config");

        let output = run_lode(workspace.path(), config_home.path(), view, &["mine"]);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{view} bad-config code: {}",
            stderr(&output)
        );
        if view == "json" {
            let payload = json_output(&output);
            assert_eq!(payload["code"], "config_error");
            assert!(payload["message"].is_string());
        } else {
            assert!(stderr(&output).contains("Could not load configuration"));
        }
        assert!(!workspace.path().join(".lode/index.db").exists());
    }
}

#[test]
fn clap_parse_errors_use_exit_two() {
    let workspace = TempDir::new().expect("workspace tempdir");
    let config_home = TempDir::new().expect("config tempdir");
    for view in VIEWS {
        let output = run_lode(
            workspace.path(),
            config_home.path(),
            view,
            &["mine", "--unknown-contract-flag"],
        );
        assert_eq!(
            output.status.code(),
            Some(2),
            "{view} clap code: {}",
            stderr(&output)
        );
    }
}
