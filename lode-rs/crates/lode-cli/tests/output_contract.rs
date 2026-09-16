use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::time::UNIX_EPOCH;

use lode_core::index::records::{FileRecord, FileStatus};
use lode_core::index::store::Store;
use lode_core::ingestion::digest::file_digest;
use lode_core::ingestion::types::Chunk;
use lode_core::relpath::WorkspacePath;
use serde_json::Value;
use tempfile::TempDir;

fn run_lode(workspace: &Path, config_home: &Path, args: &[&str]) -> Output {
    run_lode_view(workspace, config_home, "table", args)
}

fn run_lode_view(workspace: &Path, config_home: &Path, view: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_lode"))
        .arg("--workspace")
        .arg(workspace)
        .arg("--view")
        .arg(view)
        .args(args)
        .env("XDG_CONFIG_HOME", config_home)
        .env_remove("NO_COLOR")
        .env_remove("LODE_APP__OUTPUT__PALETTE")
        .env_remove("LODE_APP__OUTPUT__NO_COLOR")
        .output()
        .unwrap()
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

fn chunk(digest: &str, text: &str, seq: u32) -> Chunk {
    Chunk {
        digest: digest.to_string(),
        text: text.to_string(),
        seq,
        heading: "Heading\twith newline\n".to_string(),
        page: Some(7),
        span: None,
    }
}

#[test]
fn survey_table_is_header_only_for_an_unchanged_index() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let source = workspace.path().join("guide.txt");
    fs::write(&source, "guide").unwrap();
    fs::create_dir_all(workspace.path().join(".lode")).unwrap();
    let mut store = Store::open(
        &workspace.path().join(".lode/index.db"),
        "test-model",
        4,
        "unicode61",
    )
    .unwrap();
    store
        .replace_file(&record_for(&source, workspace.path()), "text", &[], None)
        .unwrap();
    drop(store);

    let output = run_lode(workspace.path(), config_home.path(), &["survey"]);
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "STATUS\tPATH\tFROM\tDETAIL\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn mine_table_returns_partial_failure_and_escapes_detail_fields() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    fs::write(
        workspace.path().join("lode.toml"),
        "[embedding]\nmodel = \"test-model\"\nmodel_dimension = 4\n[fts]\nstrategy = \"unicode61\"\n",
    )
    .unwrap();
    // A supported extension with invalid bytes fails in extraction before any
    // network call, making this test hermetic while exercising the summary.
    fs::write(workspace.path().join("broken.docx"), b"not a docx").unwrap();

    let output = run_lode(workspace.path(), config_home.path(), &["mine"]);
    assert_eq!(output.status.code(), Some(3));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("STATUS\tPATH\tFROM\tDETAIL"));
    let row = lines.next().expect("failed file row");
    assert_eq!(row.split('\t').count(), 4);
    assert!(row.starts_with("failed\tbroken.docx\t\t"));
    assert!(!row.contains('\n'));
}

#[test]
fn dig_table_has_one_reversible_row_per_chunk() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let source = workspace.path().join("guide.txt");
    fs::write(&source, "guide").unwrap();
    let digest = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let mut store = Store::open(
        &workspace.path().join(".lode/index.db"),
        "test-model",
        4,
        "unicode61",
    )
    .unwrap();
    store
        .replace_file(
            &record_for(&source, workspace.path()),
            "text",
            &[chunk(digest, "line one\nline two\tfield", 0)],
            None,
        )
        .unwrap();
    drop(store);

    let output = run_lode(
        workspace.path(),
        config_home.path(),
        &["dig", digest, "--radius", "0"],
    );
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next(),
        Some("SEQ\tCENTER\tSTATE\tDIGEST\tPATH\tHEADING\tPAGE\tTEXT")
    );
    let row = lines.next().expect("dig row");
    assert_eq!(row.split('\t').count(), 8);
    assert!(row.contains("\tyes\tfresh\t"));
    assert!(row.contains("\\nline two\\tfield"));
    assert!(row.contains("Heading\\twith newline\\n"));
    assert!(lines.next().is_none());
    assert!(output.stderr.is_empty());
}

#[test]
fn survey_table_preserves_rename_columns() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let old_path = workspace.path().join("old.txt");
    let new_path = workspace.path().join("new.txt");
    fs::write(&old_path, "same content").unwrap();
    fs::create_dir_all(workspace.path().join(".lode")).unwrap();
    let mut store = Store::open(
        &workspace.path().join(".lode/index.db"),
        "test-model",
        4,
        "unicode61",
    )
    .unwrap();
    store
        .replace_file(&record_for(&old_path, workspace.path()), "text", &[], None)
        .unwrap();
    drop(store);
    fs::rename(&old_path, &new_path).unwrap();

    let output = run_lode(workspace.path(), config_home.path(), &["survey"]);
    assert!(output.status.success());
    let rows = String::from_utf8_lossy(&output.stdout);
    assert_eq!(rows.lines().count(), 2);
    assert!(rows.contains("renamed\tnew.txt\told.txt\t"));
}

#[test]
fn json_execution_errors_have_stable_shape() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let missing = workspace.path().join("missing");
    let output = run_lode_view(&missing, config_home.path(), "json", &["survey"]);
    assert_eq!(output.status.code(), Some(1));
    let payload = json_output(&output);
    assert_eq!(payload["code"], "workspace_missing");
    assert!(
        payload["message"]
            .as_str()
            .unwrap()
            .contains("does not exist")
    );
    assert!(payload["hint"].as_str().unwrap().contains("--workspace"));
    assert!(output.stderr.is_empty());

    let todo = run_lode_view(workspace.path(), config_home.path(), "json", &["assay"]);
    assert_eq!(todo.status.code(), Some(1));
    let todo_payload = json_output(&todo);
    assert_eq!(todo_payload["code"], "not_implemented");
    assert_eq!(todo_payload["hint"], Value::Null);
    assert!(todo.stderr.is_empty());
}

#[test]
fn output_flags_are_documented_and_invalid_values_are_parse_errors() {
    let help = Command::new(env!("CARGO_BIN_EXE_lode"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    let help_text = String::from_utf8_lossy(&help.stdout);
    assert!(help_text.contains("--palette"));
    assert!(help_text.contains("--no-color"));
    assert!(help_text.contains("--no-progress"));
    assert!(!help_text.contains('\u{1b}'));

    let invalid = Command::new(env!("CARGO_BIN_EXE_lode"))
        .args(["survey", "--palette", "neon"])
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    assert!(invalid.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&invalid.stderr).contains('\u{1b}'));

    let workspace = TempDir::new().unwrap();
    let old_palette_name = Command::new(env!("CARGO_BIN_EXE_lode"))
        .args([
            "--workspace",
            workspace.path().to_str().unwrap(),
            "survey",
            "--palette",
            "accessible_light",
        ])
        .output()
        .unwrap();
    assert_eq!(old_palette_name.status.code(), Some(2));
}
