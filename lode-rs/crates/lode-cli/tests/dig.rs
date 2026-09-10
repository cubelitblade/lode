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

const TARGET_PREFIX: &str = "dead0001";
const OTHER_PREFIX: &str = "dead0002";

struct Fixture {
    _directory: TempDir,
    workspace: PathBuf,
    target_digest: String,
}

fn full_digest(prefix: &str) -> String {
    assert!(prefix.len() <= 64);
    format!("blake3:{prefix}{}", "0".repeat(64 - prefix.len()))
}

fn write_file(workspace: &Path, name: &str, text: &str) -> PathBuf {
    let path = workspace.join(name);
    fs::write(&path, text).unwrap();
    path
}

fn record_for(path: &Path, workspace: &Path) -> FileRecord {
    let metadata = fs::metadata(path).unwrap();
    let relative = path.strip_prefix(workspace).unwrap();
    let modified = metadata.modified().unwrap();
    let mtime = modified.duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let bytes = fs::read(path).unwrap();
    FileRecord {
        path: WorkspacePath::from_native(relative),
        digest: file_digest(&bytes),
        mtime,
        size: metadata.len(),
        status: FileStatus::Fresh,
    }
}

fn chunk(prefix: &str, text: &str, seq: u32, heading: &str) -> Chunk {
    Chunk {
        digest: full_digest(prefix),
        text: text.to_string(),
        seq,
        heading: heading.to_string(),
        page: None,
    }
}

fn fixture() -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().to_path_buf();
    let guide = write_file(&workspace, "guide.txt", "guide fixture\n");
    let other = write_file(&workspace, "other.txt", "other fixture\n");
    let index = workspace.join(".lode/index.db");

    let mut store = Store::open(&index, "test-model", 4, "unicode61").unwrap();
    store
        .replace_file(
            &record_for(&guide, &workspace),
            &[
                chunk("00000001", "before", 0, "Guide"),
                chunk(TARGET_PREFIX, "center", 1, "Guide"),
                chunk("00000002", "after", 2, "Guide"),
            ],
            None,
        )
        .unwrap();
    store
        .replace_file(
            &record_for(&other, &workspace),
            &[chunk(OTHER_PREFIX, "other", 0, "Other")],
            None,
        )
        .unwrap();
    drop(store);

    Fixture {
        _directory: directory,
        workspace,
        target_digest: full_digest(TARGET_PREFIX),
    }
}

fn run_lode(workspace: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_lode"));
    command.arg("--workspace").arg(workspace);
    command.args(args);
    command.output().unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "expected success, stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn json_output(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON ({error}); stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn dig_compact_accepts_full_digest_and_radius() {
    let fixture = fixture();
    let output = run_lode(
        &fixture.workspace,
        &["dig", &fixture.target_digest, "--radius", "1"],
    );

    assert_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Dug dead00010000"));
    assert!(!stdout.contains("0 · center"));
    assert!(stdout.contains("1 · center"));
    assert!(stdout.contains("before"));
    assert!(stdout.contains("center"));
    assert!(stdout.contains("after"));
    assert!(output.stderr.is_empty());
}

#[test]
fn get_alias_accepts_bare_digest_and_returns_json_window() {
    let fixture = fixture();
    let bare_digest = fixture.target_digest.strip_prefix("blake3:").unwrap();
    let output = run_lode(
        &fixture.workspace,
        &["--view", "json", "get", bare_digest, "--radius", "1"],
    );

    assert_success(&output);
    let payload = json_output(&output);
    assert_eq!(payload["digest"], fixture.target_digest);
    assert_eq!(payload["window"]["center_seq"], 1);
    assert_eq!(payload["window"]["radius"], 1);
    let chunks = payload["window"]["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0]["text"], "before");
    assert_eq!(chunks[1]["text"], "center");
    assert_eq!(chunks[2]["text"], "after");
    assert_eq!(chunks[1]["paths"][0]["path"], "guide.txt");
}

#[test]
fn dig_json_reports_invalid_not_found_and_ambiguous_inputs() {
    let fixture = fixture();

    let invalid = run_lode(&fixture.workspace, &["--view", "json", "dig", "not-hex"]);
    assert!(!invalid.status.success());
    assert_eq!(json_output(&invalid)["code"], "invalid_digest");
    assert!(invalid.stderr.is_empty());

    let missing_digest = full_digest("beef");
    let missing = run_lode(
        &fixture.workspace,
        &["--view", "json", "dig", &missing_digest],
    );
    assert!(!missing.status.success());
    assert_eq!(json_output(&missing)["code"], "not_found");

    let ambiguous = run_lode(&fixture.workspace, &["--view", "json", "dig", "dead"]);
    assert!(!ambiguous.status.success());
    let payload = json_output(&ambiguous);
    assert_eq!(payload["code"], "ambiguous");
    let candidates = payload["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 2);
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.get("text").is_none())
    );
}

#[test]
fn dig_json_reports_no_index_without_touching_embedding() {
    let directory = tempfile::tempdir().unwrap();
    let output = run_lode(directory.path(), &["--view", "json", "dig", "dead0001"]);

    assert!(!output.status.success());
    assert_eq!(json_output(&output)["code"], "no_index");
    assert!(output.stderr.is_empty());
}
