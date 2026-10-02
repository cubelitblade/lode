use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

use lode_core::index::records::{FileRecord, FileStatus};
use lode_core::index::store::Store;
use lode_core::ingestion::digest::file_digest;
use lode_core::ingestion::types::Chunk;
use lode_core::relpath::WorkspacePath;
use serde_json::Value;
use tempfile::{TempDir, tempdir};

const DIGEST: &str = "blake3:dead000100000000000000000000000000000000000000000000000000000000";

struct Fixture {
    _directory: TempDir,
    workspace: PathBuf,
    config_home: TempDir,
    cache_home: TempDir,
    digest: String,
}

struct MockEmbeddingServer {
    endpoint: String,
    thread: thread::JoinHandle<()>,
}

impl MockEmbeddingServer {
    fn join(self) {
        self.thread.join().expect("mock server thread panicked");
    }
}

fn record_for(path: &Path, workspace: &Path) -> FileRecord {
    let metadata = fs::metadata(path).unwrap();
    let mtime = metadata
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    FileRecord {
        path: WorkspacePath::from_native(path.strip_prefix(workspace).unwrap()),
        digest: file_digest(&fs::read(path).unwrap()),
        mtime,
        size: metadata.len(),
        status: FileStatus::Fresh,
    }
}

fn fixture(config: &str, text: &str) -> Fixture {
    fixture_with_physical_tokenizer(config, text, "unicode61")
}

fn fixture_with_physical_tokenizer(config: &str, text: &str, physical_tokenizer: &str) -> Fixture {
    let directory = tempdir().unwrap();
    let workspace = directory.path().to_path_buf();
    let config_home = tempdir().unwrap();
    let cache_home = tempdir().unwrap();
    fs::write(workspace.join("lode.toml"), config).unwrap();
    let source = workspace.join("notes.txt");
    fs::write(&source, text).unwrap();
    fs::create_dir_all(workspace.join(".lode")).unwrap();

    let mut store = Store::open(
        &workspace.join(".lode/index.db"),
        "test-model",
        4,
        physical_tokenizer,
    )
    .unwrap();
    store
        .replace_file(
            &record_for(&source, &workspace),
            "text",
            &[Chunk {
                digest: DIGEST.to_string(),
                text: text.to_string(),
                seq: 0,
                heading: "Notes".to_string(),
                page: None,
                span: None,
            }],
            Some(&[vec![1.0, 0.0, 0.0, 0.0]]),
        )
        .unwrap();

    Fixture {
        _directory: directory,
        workspace,
        config_home,
        cache_home,
        digest: DIGEST.to_string(),
    }
}

fn run_lode(fixture: &Fixture, view: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_lode"))
        .arg("--workspace")
        .arg(&fixture.workspace)
        .arg("--view")
        .arg(view)
        .args(args)
        .env("XDG_CONFIG_HOME", fixture.config_home.path())
        .env("XDG_CACHE_HOME", fixture.cache_home.path())
        .env("LODE_CACHE_DIR", fixture.cache_home.path())
        .env_remove("NO_COLOR")
        .output()
        .unwrap()
}

fn json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON ({error}); stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn config(endpoint: &str) -> String {
    format!(
        "[embedding]\nprovider = \"openai_compatible\"\nmodel = \"test-model\"\nmodel_dimension = 4\n[embedding.openai_compatible]\nendpoint = \"{endpoint}\"\nmax_retries = 0\n[fts]\nstrategy = \"unicode61\"\n"
    )
}

fn spawn_embedding_server(expected_requests: usize) -> MockEmbeddingServer {
    spawn_embedding_server_with_dimension(expected_requests, 4)
}

fn spawn_embedding_server_with_dimension(
    expected_requests: usize,
    dimension: usize,
) -> MockEmbeddingServer {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let thread = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut handled = 0;
        while handled < expected_requests && Instant::now() < deadline {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .unwrap();
                    let (path, _) = read_request(&mut stream);
                    assert_eq!(path, "/v1/embeddings");
                    let mut embedding = vec![0.0; dimension];
                    embedding[0] = 1.0;
                    let body = serde_json::json!({
                        "data": [{"embedding": embedding, "index": 0}],
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                    handled += 1;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept mock request: {error}"),
            }
        }
        assert_eq!(handled, expected_requests, "mock server timed out");
    });
    MockEmbeddingServer { endpoint, thread }
}

fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let header_end = loop {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "client closed before sending headers");
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
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    while request.len() < header_end + content_length {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "client closed before sending body");
        request.extend_from_slice(&buffer[..count]);
    }
    let path = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap()
        .to_string();
    (
        path,
        request[header_end..header_end + content_length].to_vec(),
    )
}

#[test]
fn why_json_matches_prospect_score_and_supports_analyze_alias() {
    let server = spawn_embedding_server(2);
    let fixture = fixture(&config(&server.endpoint), "hello indexed world");

    let prospect = run_lode(&fixture, "json", &["prospect", "hello", "--top-k", "2"]);
    assert!(
        prospect.status.success(),
        "prospect stderr={:?}",
        prospect.stderr
    );
    let prospect_json = json(&prospect);
    let expected_score = prospect_json["hits"][0]["score"].as_f64().unwrap();

    let why = run_lode(
        &fixture,
        "json",
        &[
            "analyze",
            "why",
            fixture.digest.strip_prefix("blake3:").unwrap(),
            "hello",
        ],
    );
    assert!(why.status.success(), "why stderr={:?}", why.stderr);
    assert!(why.stderr.is_empty());
    let payload = json(&why);
    assert_eq!(payload.as_object().unwrap().len(), 15);
    assert_eq!(payload["query"], "hello");
    assert_eq!(payload["top_k"], 10);
    assert_eq!(payload["digest"], fixture.digest);
    assert_eq!(payload["paths"][0]["path"], "notes.txt");
    assert_eq!(payload["paths"][0]["state"], "fresh");
    assert_eq!(payload["heading"], "Notes");
    assert_eq!(payload["has_stale"], false);
    assert_eq!(payload["sources"]["semantic"]["status"], "matched");
    assert_eq!(payload["sources"]["lexical"]["status"], "matched");
    assert_eq!(payload["norm"]["name"], "softmax");
    assert_eq!(payload["fusion"]["name"], "linear");
    assert_eq!(
        payload["ranking_factors"]["semantic"]["metric"],
        "contribution"
    );
    assert_eq!(payload["rank"], 1);
    assert_eq!(payload["in_results"], true);
    assert!((payload["combined"].as_f64().unwrap() - expected_score).abs() < 1e-12);
    server.join();
}

#[test]
fn why_human_views_and_table_share_the_json_explanation() {
    let server = spawn_embedding_server(3);
    let fixture = fixture(&config(&server.endpoint), "hello indexed world");
    let second = fixture.workspace.join("other.txt");
    fs::write(&second, "hello indexed world").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert!(
        store
            .reference_file(&record_for(&second, &fixture.workspace), "text")
            .unwrap()
    );
    drop(store);

    let compact = run_lode(
        &fixture,
        "compact",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert!(
        compact.status.success(),
        "compact stderr={:?}",
        compact.stderr
    );
    let compact_stdout = String::from_utf8_lossy(&compact.stdout);
    assert!(compact_stdout.contains("Assay why"));
    assert!(compact_stdout.contains("Query: hello"));
    assert!(compact_stdout.contains("Score:"));
    assert!(compact_stdout.contains("semantic: matched"));
    assert!(compact_stdout.contains("lexical: matched"));
    assert!(!compact_stdout.contains("{\""));
    assert!(compact.stderr.is_empty());

    let extended = run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert!(
        extended.status.success(),
        "extended stderr={:?}",
        extended.stderr
    );
    let extended_stdout = String::from_utf8_lossy(&extended.stdout);
    assert!(extended_stdout.contains("Target:"));
    assert!(extended_stdout.contains("Sources:"));
    assert!(extended_stdout.contains("Plan:"));
    assert!(extended_stdout.contains("Fusion explanation:"));
    assert!(extended_stdout.contains("raw score:"));
    assert!(extended_stdout.contains("prepared score:"));
    assert!(extended_stdout.contains("values (display rounded):"));
    assert!(extended_stdout.contains("other.txt (fresh)"));

    let table = run_lode(
        &fixture,
        "table",
        &["assay", "why", &fixture.digest, "hello\tworld"],
    );
    assert!(table.status.success(), "table stderr={:?}", table.stderr);
    assert!(table.stderr.is_empty());
    let table_stdout = String::from_utf8_lossy(&table.stdout);
    let rows = table_stdout
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].len(), 23);
    assert_eq!(
        rows[0],
        vec![
            "QUERY",
            "TOP_K",
            "DIGEST",
            "PATH",
            "HEADING",
            "PAGE",
            "SEQ",
            "STATE",
            "COMBINED",
            "RANK",
            "IN_RESULTS",
            "SOURCE",
            "STATUS",
            "POOL_SIZE",
            "RAW_SCORE",
            "PREPARED_SCORE",
            "POOL_RANK",
            "FACTOR_METRIC",
            "FACTOR_VALUE",
            "NORM",
            "NORM_PARAMS",
            "FUSION",
            "FUSION_PARAMS",
        ]
    );
    assert_eq!(rows[1][11], "semantic");
    assert_eq!(rows[2][11], "lexical");
    assert_eq!(rows[1][0], "hello\\tworld");
    assert_eq!(rows[1][12], "matched");
    assert_eq!(rows[2][12], "matched");
    assert_eq!(rows[1][10], "true");
    assert_eq!(rows[2][10], "true");
    assert_eq!(rows[1][19], "softmax");
    assert_eq!(rows[1][21], "linear");
    server.join();
}

#[test]
fn why_human_does_not_call_a_negative_score_zero() {
    let server = spawn_embedding_server(1);
    let fixture = fixture(
        &format!(
            "{}[retrieval.fusion.linear]\nsemantic_factor = -1.0\nlexical_factor = 0.0\n",
            config(&server.endpoint)
        ),
        "hello indexed world",
    );

    let output = run_lode(
        &fixture,
        "compact",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("not ranked (no positive score)"));
    assert!(!stdout.contains("not ranked (zero score)"));
    server.join();
}

#[test]
fn why_human_and_table_views_warn_for_stale_snapshots() {
    let server = spawn_embedding_server(2);
    let fixture = fixture(&config(&server.endpoint), "old indexed text");
    fs::write(fixture.workspace.join("notes.txt"), "new workspace text").unwrap();

    let compact = run_lode(
        &fixture,
        "compact",
        &["assay", "why", &fixture.digest, "old"],
    );
    assert!(
        compact.status.success(),
        "compact stderr={:?}",
        compact.stderr
    );
    assert!(String::from_utf8_lossy(&compact.stdout).contains("[stale]"));
    assert_eq!(
        String::from_utf8_lossy(&compact.stderr)
            .matches("stale indexed snapshot")
            .count(),
        1
    );

    let table = run_lode(&fixture, "table", &["assay", "why", &fixture.digest, "old"]);
    assert!(table.status.success(), "table stderr={:?}", table.stderr);
    assert_eq!(String::from_utf8_lossy(&table.stdout).lines().count(), 3);
    assert_eq!(
        String::from_utf8_lossy(&table.stderr)
            .matches("stale indexed snapshot")
            .count(),
        1
    );
    server.join();
}

#[test]
fn why_extended_marks_rrf_normalization_as_skipped() {
    let server = spawn_embedding_server(1);
    let fixture = fixture(
        &format!(
            "{}[retrieval.fusion]\ntype = \"rrf\"\n[retrieval.fusion.rrf]\nk = 11\n",
            config(&server.endpoint)
        ),
        "hello indexed world",
    );

    let output = run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("normalization: none (skipped by RRF) params={}"));
    assert!(stdout.contains("fusion: rrf params={\"k\":11}"));
    assert!(stdout.contains("factor: rank = 1"));
    assert!(!stdout.contains("factor: rank = 1.0000"));
    server.join();
}

#[test]
fn why_json_keeps_a_positive_target_rank_beyond_top_k() {
    let server = spawn_embedding_server(1);
    let fixture = fixture(
        &format!(
            "{}[retrieval.fusion.linear]\nsemantic_factor = 1.0\nlexical_factor = 0.0\n",
            config(&server.endpoint)
        ),
        "primary indexed world",
    );
    let second = fixture.workspace.join("second.txt");
    fs::write(&second, "secondary indexed world").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    store
        .replace_file(
            &record_for(&second, &fixture.workspace),
            "text",
            &[Chunk {
                digest: "blake3:dead000200000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                text: "secondary indexed world".to_string(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            Some(&[vec![0.8, 0.6, 0.0, 0.0]]),
        )
        .unwrap();

    let target = "blake3:dead000200000000000000000000000000000000000000000000000000000000";
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", target, "primary", "--top-k", "1"],
    );
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let payload = json(&output);
    assert_eq!(payload["sources"]["semantic"]["status"], "matched");
    assert_eq!(payload["rank"], 2);
    assert_eq!(payload["in_results"], false);
    assert!(payload["combined"].as_f64().unwrap() > 0.0);
    server.join();
}

#[test]
fn why_json_reports_rrf_parameters_and_positional_rank_factors() {
    let server = spawn_embedding_server(1);
    let fixture = fixture(
        &format!(
            "{}[retrieval.fusion]\ntype = \"rrf\"\n[retrieval.fusion.rrf]\nk = 11\n",
            config(&server.endpoint)
        ),
        "hello indexed world",
    );
    let second = fixture.workspace.join("second.txt");
    fs::write(&second, "a different physical snapshot").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    store
        .replace_file(
            &record_for(&second, &fixture.workspace),
            "text",
            &[Chunk {
                digest: "blake3:dead000200000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                text: "hello indexed world".to_string(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            Some(&[vec![1.0, 0.0, 0.0, 0.0]]),
        )
        .unwrap();
    drop(store);

    let target = "blake3:dead000200000000000000000000000000000000000000000000000000000000";
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", target, "hello", "--top-k", "1"],
    );
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let payload = json(&output);
    assert_eq!(payload["top_k"], 1);
    assert_eq!(
        payload["norm"],
        serde_json::json!({
            "name": null,
            "params": {},
        })
    );
    assert_eq!(
        payload["fusion"],
        serde_json::json!({
            "name": "rrf",
            "params": {"k": 11},
        })
    );
    assert_eq!(payload["ranking_factors"]["semantic"]["metric"], "rank");
    assert_eq!(payload["ranking_factors"]["lexical"]["metric"], "rank");
    assert_eq!(payload["sources"]["semantic"]["pool_rank"], 2);
    assert_eq!(payload["sources"]["lexical"]["pool_rank"], 2);
    assert_eq!(
        payload["ranking_factors"]["semantic"]["value"]
            .as_f64()
            .unwrap(),
        2.0
    );
    assert_eq!(
        payload["ranking_factors"]["lexical"]["value"]
            .as_f64()
            .unwrap(),
        2.0
    );
    assert_eq!(payload["rank"], 2);
    assert_eq!(payload["in_results"], false);
    assert!((payload["combined"].as_f64().unwrap() - 2.0 / 13.0).abs() < 1e-12);
    server.join();
}

#[test]
fn why_json_refreshes_stale_references_but_keeps_the_old_snapshot() {
    let server = spawn_embedding_server(1);
    let fixture = fixture(&config(&server.endpoint), "old indexed text");
    fs::write(fixture.workspace.join("notes.txt"), "new workspace text").unwrap();

    let output = run_lode(&fixture, "json", &["assay", "why", &fixture.digest, "old"]);
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let payload = json(&output);
    assert_eq!(payload["has_stale"], true);
    assert_eq!(payload["paths"][0]["state"], "stale");

    let store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    let lookup = store.get_chunk_window("dead0001", 0).unwrap();
    let lode_core::index::records::ChunkLookup::Found(window) = lookup else {
        panic!("expected the old chunk snapshot");
    };
    assert_eq!(window.target.text, "old indexed text");
    server.join();
}

#[test]
fn why_json_reports_statuses_and_plan_parameters_without_embedding_when_disabled() {
    let fixture = fixture(
        &format!(
            "{}[retrieval.norm]\ntype = \"softmax\"\n[retrieval.norm.softmax]\ntemperature = 2.5\n[retrieval.fusion.linear]\nsemantic_factor = 0.0\nlexical_factor = 1.0\n",
            config("http://127.0.0.1:1")
        ),
        "hello indexed world",
    );
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "missing"],
    );
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let payload = json(&output);
    assert_eq!(payload["sources"]["semantic"]["status"], "disabled");
    assert_eq!(payload["sources"]["semantic"]["raw_score"], Value::Null);
    assert_eq!(payload["sources"]["lexical"]["status"], "empty");
    assert_eq!(payload["sources"]["lexical"]["pool_size"], 0);
    assert_eq!(
        payload["norm"],
        serde_json::json!({
            "name": "softmax",
            "params": {"temperature": 2.5},
        })
    );
    assert_eq!(payload["fusion"]["name"], "linear");
    assert_eq!(payload["rank"], Value::Null);
    assert_eq!(payload["in_results"], false);
}

#[test]
fn why_json_reports_not_retrieved_source_status() {
    let fixture = fixture(
        &format!(
            "{}[retrieval.fusion.linear]\nsemantic_factor = 0.0\nlexical_factor = 1.0\n",
            config("http://127.0.0.1:1")
        ),
        "hello indexed world",
    );
    let second = fixture.workspace.join("second.txt");
    fs::write(&second, "unrelated indexed world").unwrap();
    let target = "blake3:dead000200000000000000000000000000000000000000000000000000000000";
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    store
        .replace_file(
            &record_for(&second, &fixture.workspace),
            "text",
            &[Chunk {
                digest: target.to_owned(),
                text: "unrelated indexed world".to_owned(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            Some(&[vec![0.0, 1.0, 0.0, 0.0]]),
        )
        .unwrap();
    drop(store);

    let output = run_lode(&fixture, "json", &["assay", "why", target, "hello"]);
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    let payload = json(&output);
    assert_eq!(payload["sources"]["semantic"]["status"], "disabled");
    assert_eq!(payload["sources"]["lexical"]["status"], "not_retrieved");
    assert_eq!(payload["sources"]["lexical"]["pool_size"], 1);
    assert_eq!(payload["sources"]["lexical"]["raw_score"], Value::Null);
    assert_eq!(payload["rank"], Value::Null);
    assert_eq!(payload["in_results"], false);
}

#[test]
fn why_json_explains_all_fts_strategies_without_semantic_requests() {
    for (strategy, physical, text, query) in [
        ("unicode61", "unicode61", "hello rocks", "hello"),
        ("trigram", "trigram", "矿石探矿", "矿石探"),
        ("simple", "simple", "知识", "知识"),
        ("jieba", "simple", "知识", "知识"),
    ] {
        let config = format!(
            "[fts]\nstrategy = \"{strategy}\"\n[retrieval.fusion.linear]\nsemantic_factor = 0.0\nlexical_factor = 1.0\n"
        );
        let fixture = fixture_with_physical_tokenizer(&config, text, physical);
        let output = run_lode(&fixture, "json", &["assay", "why", &fixture.digest, query]);
        assert!(
            output.status.success(),
            "{strategy}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let payload = json(&output);
        assert_eq!(payload["digest"], fixture.digest);
        assert_eq!(payload["sources"]["semantic"]["status"], "disabled");
        assert_eq!(payload["sources"]["lexical"]["status"], "matched");
        assert_eq!(payload["rank"], 1);
    }
}

#[test]
fn why_json_reports_min_max_parameters() {
    let fixture = fixture(
        &format!(
            "{}[retrieval.norm]\ntype = \"minmax\"\n[retrieval.fusion.linear]\nsemantic_factor = 0.0\nlexical_factor = 1.0\n",
            config("http://127.0.0.1:1")
        ),
        "hello indexed world",
    );
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert!(output.status.success(), "why stderr={:?}", output.stderr);
    assert_eq!(
        json(&output)["norm"],
        serde_json::json!({
            "name": "min-max",
            "params": {},
        })
    );
}

#[test]
fn why_json_maps_embedding_and_dimension_failures() {
    let unavailable = fixture(&config("http://127.0.0.1:1"), "hello indexed world");
    let embedding_error = run_lode(
        &unavailable,
        "json",
        &["assay", "why", &unavailable.digest, "hello"],
    );
    assert_eq!(embedding_error.status.code(), Some(1));
    assert_eq!(json(&embedding_error)["code"], "embedding_error");
    assert!(embedding_error.stderr.is_empty());

    let server = spawn_embedding_server_with_dimension(1, 3);
    let dimension_config = config(&server.endpoint);
    let mismatch = fixture(&dimension_config, "hello indexed world");
    let dimension_error = run_lode(
        &mismatch,
        "json",
        &["assay", "why", &mismatch.digest, "hello"],
    );
    assert_eq!(dimension_error.status.code(), Some(1));
    assert_eq!(json(&dimension_error)["code"], "dimension_error");
    server.join();
}

#[test]
fn why_validates_before_index_and_reports_human_errors_without_json() {
    let fixture = fixture(&config("http://127.0.0.1:1"), "hello");

    let empty = run_lode(&fixture, "json", &["assay", "why", &fixture.digest, "  "]);
    assert_eq!(empty.status.code(), Some(1));
    assert_eq!(json(&empty)["code"], "invalid_query");

    let invalid = run_lode(&fixture, "json", &["assay", "why", "not-hex", "hello"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert_eq!(json(&invalid)["code"], "invalid_digest");

    let zero = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "hello", "--top-k", "0"],
    );
    assert_eq!(zero.status.code(), Some(1));
    assert_eq!(json(&zero)["code"], "invalid_top_k");

    let missing = run_lode(&fixture, "json", &["assay", "why", "beef", "hello"]);
    assert_eq!(missing.status.code(), Some(1));
    assert_eq!(json(&missing)["code"], "not_found");

    let human = run_lode(
        &fixture,
        "compact",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert_eq!(human.status.code(), Some(1));
    assert!(human.stdout.is_empty());
    let human_stderr = String::from_utf8_lossy(&human.stderr);
    assert!(human_stderr.contains("Could not explain the retrieval score"));
    assert!(!human_stderr.contains("{\"code\""));

    fs::write(
        fixture.workspace.join("lode.toml"),
        "[fts]\nstrategy = \"trigram\"\n",
    )
    .unwrap();
    let mismatch = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "hello"],
    );
    assert_eq!(mismatch.status.code(), Some(1));
    assert_eq!(json(&mismatch)["code"], "index_open_error");
}

#[test]
fn why_json_reports_ambiguous_digest_candidates() {
    let fixture = fixture(&config("http://127.0.0.1:1"), "hello");
    let second = fixture.workspace.join("other.txt");
    fs::write(&second, "second").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    store
        .replace_file(
            &record_for(&second, &fixture.workspace),
            "text",
            &[Chunk {
                digest: "blake3:dead000200000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                text: "second".to_string(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            Some(&[vec![1.0, 0.0, 0.0, 0.0]]),
        )
        .unwrap();
    drop(store);

    let output = run_lode(&fixture, "json", &["assay", "why", "dead", "hello"]);
    assert_eq!(output.status.code(), Some(1));
    let payload = json(&output);
    assert_eq!(payload["code"], "ambiguous");
    assert_eq!(payload["candidates"].as_array().unwrap().len(), 2);
    assert!(payload["candidates"][0].get("text").is_none());

    let human = run_lode(&fixture, "compact", &["assay", "why", "dead", "hello"]);
    assert_eq!(human.status.code(), Some(1));
    assert!(human.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&human.stderr);
    assert!(stderr.contains("other.txt"));
    assert!(stderr.contains("dead0002"));
}

#[path = "assay_why/fts_details.rs"]
mod fts_details;

#[path = "assay_why/fts_extended.rs"]
mod fts_extended;
