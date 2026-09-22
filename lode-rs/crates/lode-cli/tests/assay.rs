use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::UNIX_EPOCH;

use lode_core::index::records::{FileRecord, FileStatus};
use lode_core::index::store::Store;
use lode_core::ingestion::digest::file_digest;
use lode_core::ingestion::types::Chunk;
use lode_core::relpath::WorkspacePath;
use serde_json::Value;
use tempfile::TempDir;

struct Fixture {
    _directory: TempDir,
    workspace: PathBuf,
    config_home: TempDir,
    cache_home: TempDir,
    digest: String,
}

fn full_digest(prefix: &str) -> String {
    format!("blake3:{prefix}{}", "0".repeat(64 - prefix.len()))
}

fn record_for(path: &Path, workspace: &Path) -> FileRecord {
    let metadata = fs::metadata(path).unwrap();
    let modified = metadata.modified().unwrap();
    let mtime = modified.duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    FileRecord {
        path: WorkspacePath::from_native(path.strip_prefix(workspace).unwrap()),
        digest: file_digest(&fs::read(path).unwrap()),
        mtime,
        size: metadata.len(),
        status: FileStatus::Fresh,
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
        .env_remove("LODE_APP__OUTPUT__PALETTE")
        .env_remove("LODE_APP__OUTPUT__NO_COLOR")
        .output()
        .unwrap()
}

fn json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON ({error}); stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn fixture(strategy: &str, text: &str) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().to_path_buf();
    let config_home = tempfile::tempdir().unwrap();
    let cache_home = tempfile::tempdir().unwrap();
    fs::write(
        workspace.join("lode.toml"),
        format!("[fts]\nstrategy = \"{strategy}\"\n"),
    )
    .unwrap();
    let source = workspace.join("notes.txt");
    fs::write(&source, text).unwrap();
    fs::create_dir_all(workspace.join(".lode")).unwrap();

    // `jieba` shares the physical `simple` tokenizer. Creating this fixture
    // as `simple` avoids configuring the query-only dictionary in the parent
    // test process; the CLI child still opens it with the requested strategy.
    let physical = if strategy == "jieba" {
        "simple"
    } else {
        strategy
    };
    let digest = full_digest("dead0001");
    let mut store =
        Store::open(&workspace.join(".lode/index.db"), "test-model", 4, physical).unwrap();
    store
        .replace_file(
            &record_for(&source, &workspace),
            "text",
            &[Chunk {
                digest: digest.clone(),
                text: text.to_string(),
                seq: 0,
                heading: "Notes".to_string(),
                page: None,
                span: None,
            }],
            None,
        )
        .unwrap();

    Fixture {
        _directory: directory,
        workspace,
        config_home,
        cache_home,
        digest,
    }
}

#[test]
fn how_json_supports_all_strategies_and_variants() {
    for (strategy, text, physical, expected_token) in [
        ("unicode61", "Café Rocks", "unicode61", "cafe"),
        ("trigram", "矿石探矿", "trigram", "矿石探"),
        ("simple", "知识", "simple", "知"),
        ("jieba", "知识", "simple", "知"),
    ] {
        let fixture = fixture(strategy, text);
        let output = run_lode(&fixture, "json", &["assay", "how", &fixture.digest]);
        assert!(
            output.status.success(),
            "{strategy}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let payload = json(&output);
        assert_eq!(payload["digest"], fixture.digest);
        assert_eq!(payload["tokenizer"]["strategy"], strategy);
        assert_eq!(payload["tokenizer"]["tokenize"], physical);
        assert_eq!(payload["text"], text);
        assert_eq!(payload["has_stale"], false);
        assert_eq!(payload["tokens"][0]["surface"], expected_token);
        assert!(payload["terms"].is_array());
        if matches!(strategy, "simple" | "jieba") {
            assert!(
                !payload["tokens"][0]["variants"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        } else {
            assert_eq!(payload["tokens"][0]["variants"], serde_json::json!([]));
        }
    }
}

#[test]
fn how_json_accepts_bare_digest_and_analyze_alias() {
    let fixture = fixture("unicode61", "hello world");
    let bare = fixture.digest.strip_prefix("blake3:").unwrap();
    let output = run_lode(&fixture, "json", &["analyze", "how", bare]);
    assert!(output.status.success());
    assert_eq!(json(&output)["digest"], fixture.digest);
}

#[test]
fn unsupported_view_fails_before_index_or_stale_changes() {
    let fixture = fixture("unicode61", "hello");
    fs::write(
        fixture.workspace.join("notes.txt"),
        "changed workspace text",
    )
    .unwrap();

    for view in ["compact", "extended", "table"] {
        let output = run_lode(&fixture, view, &["assay", "how", &fixture.digest]);
        assert_eq!(output.status.code(), Some(1), "view={view}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("only `--view json`"),
            "view={view}: stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "view={view}");
    }

    let store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert_eq!(
        store
            .get_file(&WorkspacePath::from_posix("notes.txt"))
            .unwrap()
            .unwrap()
            .status,
        FileStatus::Fresh
    );
}

#[test]
fn how_json_reports_stale_snapshot_without_reindexing() {
    let fixture = fixture("unicode61", "old indexed text");
    fs::write(fixture.workspace.join("notes.txt"), "new workspace text").unwrap();
    let output = run_lode(&fixture, "json", &["assay", "how", &fixture.digest]);
    assert!(output.status.success());
    let payload = json(&output);
    assert_eq!(payload["text"], "old indexed text");
    assert_eq!(payload["has_stale"], true);
    assert_eq!(payload["paths"][0]["state"], "stale");

    let store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert_eq!(
        store
            .get_file(&WorkspacePath::from_posix("notes.txt"))
            .unwrap()
            .unwrap()
            .status,
        FileStatus::Stale
    );
}

#[test]
fn how_json_reports_deleted_source_as_stale_snapshot() {
    let fixture = fixture("unicode61", "deleted source text");
    fs::remove_file(fixture.workspace.join("notes.txt")).unwrap();

    let output = run_lode(&fixture, "json", &["assay", "how", &fixture.digest]);
    assert!(output.status.success());
    let payload = json(&output);
    assert_eq!(payload["text"], "deleted source text");
    assert_eq!(payload["has_stale"], true);
    assert_eq!(payload["paths"][0]["path"], "notes.txt");
    assert_eq!(payload["paths"][0]["state"], "stale");

    let store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert_eq!(
        store
            .get_file(&WorkspacePath::from_posix("notes.txt"))
            .unwrap()
            .unwrap()
            .status,
        FileStatus::Stale
    );
    let bare = fixture.digest.strip_prefix("blake3:").unwrap();
    let lookup = store.get_chunk_window(bare, 0).unwrap();
    match lookup {
        lode_core::index::records::ChunkLookup::Found(window) => {
            assert_eq!(window.target.text, "deleted source text");
        }
        other => panic!("expected the stale chunk to remain indexed, got {other:?}"),
    }
}

#[test]
fn how_json_reports_no_index_and_invalid_digest() {
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().to_path_buf();
    let config_home = tempfile::tempdir().unwrap();
    let cache_home = tempfile::tempdir().unwrap();
    let missing_fixture = Fixture {
        _directory: directory,
        workspace,
        config_home,
        cache_home,
        digest: String::new(),
    };
    let no_index = run_lode(&missing_fixture, "json", &["assay", "how", "dead"]);
    assert_eq!(no_index.status.code(), Some(1));
    assert_eq!(json(&no_index)["code"], "no_index");

    let existing = fixture("unicode61", "hello");
    let invalid = run_lode(&existing, "json", &["assay", "how", "not-hex"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert_eq!(json(&invalid)["code"], "invalid_digest");
}

#[test]
fn how_json_reports_ambiguous_prefix_without_chunk_text() {
    let fixture = fixture("unicode61", "hello");
    let source = fixture.workspace.join("other.txt");
    fs::write(&source, "second").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    store
        .replace_file(
            &record_for(&source, &fixture.workspace),
            "text",
            &[Chunk {
                digest: full_digest("dead0002"),
                text: "second".to_string(),
                seq: 1,
                heading: "Notes".to_string(),
                page: None,
                span: None,
            }],
            None,
        )
        .unwrap();
    drop(store);

    let output = run_lode(&fixture, "json", &["assay", "how", "dead"]);
    assert_eq!(output.status.code(), Some(1));
    let payload = json(&output);
    assert_eq!(payload["code"], "ambiguous");
    assert_eq!(payload["candidates"].as_array().unwrap().len(), 2);
    assert!(payload["candidates"][0].get("text").is_none());
}

#[test]
fn how_json_rejects_a_physical_tokenizer_mismatch() {
    let fixture = fixture("unicode61", "hello");
    fs::write(
        fixture.workspace.join("lode.toml"),
        "[fts]\nstrategy = \"trigram\"\n",
    )
    .unwrap();
    let output = run_lode(&fixture, "json", &["assay", "how", &fixture.digest]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(json(&output)["code"], "index_open_error");
}
