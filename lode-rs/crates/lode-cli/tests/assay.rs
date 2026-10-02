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
                heading: "Notes\twith newline\n".to_string(),
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

fn settings_fixture(strategy: &str) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().to_path_buf();
    let config_home = tempfile::tempdir().unwrap();
    let cache_home = tempfile::tempdir().unwrap();
    fs::write(
        workspace.join("lode.toml"),
        format!("[fts]\nstrategy = \"{strategy}\"\n"),
    )
    .unwrap();
    Fixture {
        _directory: directory,
        workspace,
        config_home,
        cache_home,
        digest: String::new(),
    }
}

#[test]
fn debug_tokens_json_supports_all_strategies_and_variants() {
    for (strategy, text, physical, expected_token) in [
        ("unicode61", "Café Rocks", "unicode61", "cafe"),
        ("trigram", "矿石探矿", "trigram", "矿石探"),
        ("simple", "知识", "simple", "知"),
        ("jieba", "知识", "simple", "知"),
    ] {
        let fixture = fixture(strategy, text);
        let output = run_lode(&fixture, "json", &["debug", "tokens", &fixture.digest]);
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
fn debug_tokens_json_accepts_bare_digest() {
    let fixture = fixture("unicode61", "hello world");
    let bare = fixture.digest.strip_prefix("blake3:").unwrap();
    let output = run_lode(&fixture, "json", &["debug", "tokens", bare]);
    assert!(output.status.success());
    assert_eq!(json(&output)["digest"], fixture.digest);
}

#[test]
fn how_query_does_not_require_an_index() {
    for (strategy, query) in [
        ("unicode61", "知识 rocks"),
        ("trigram", "矿石探矿"),
        ("simple", "知识"),
        ("jieba", "中华人民共和国"),
    ] {
        let fixture = settings_fixture(strategy);
        let output = run_lode(&fixture, "json", &["assay", "how", query]);
        assert!(
            output.status.success(),
            "{strategy}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let payload = json(&output);
        assert_eq!(payload["query"], query);
        assert_eq!(payload["tokenizer"]["strategy"], strategy);
        assert_eq!(
            payload["tokenizer"]["tokenize"],
            if strategy == "jieba" {
                "simple"
            } else {
                strategy
            }
        );
        if strategy == "simple" {
            assert_eq!(payload["helper"], "simple_query");
        } else if strategy == "jieba" {
            assert_eq!(payload["helper"], "jieba_query");
        } else {
            assert_eq!(payload["helper"], Value::Null);
        }
        if strategy != "trigram" {
            assert!(!payload["match_expression"].as_str().unwrap().is_empty());
        }
    }
}

#[test]
fn how_trigram_expression_executes_like_sparse_search() {
    for query in [
        "alpha \" beta",
        "a\"b",
        "a\"\"b",
        "\"\"\"",
        "a*b",
        "a:b",
        "a\nb",
        "a\tb",
        "a\rb",
        "a\u{1f}b",
        "矿\"a",
        "ab矿",
        "ab",
        "矿石",
    ] {
        let fixture = fixture("trigram", query);
        let output = run_lode(&fixture, "json", &["assay", "how", query]);
        assert!(output.status.success(), "{query:?}: {output:?}");
        assert!(output.stderr.is_empty());
        let payload = json(&output);
        assert_eq!(payload["query"], query);
        let expression = payload["match_expression"].as_str().unwrap();
        let generated = lode_core::fts::match_query(query, "trigram").unwrap();
        assert_eq!(
            generated,
            lode_core::fts::MatchExpr::Prebuilt(expression.to_string())
        );
        if query.chars().count() < 3 {
            assert!(expression.is_empty(), "{query:?}");
            continue;
        }

        let store = Store::open(
            &fixture.workspace.join(".lode/index.db"),
            "test-model",
            4,
            "trigram",
        )
        .unwrap();
        let hits = store.sparse_search(&generated, 5).unwrap();
        let displayed_hits = store
            .sparse_search(
                &lode_core::fts::MatchExpr::Prebuilt(expression.to_string()),
                5,
            )
            .unwrap();
        assert_eq!(hits.len(), 1, "{query:?}");
        assert_eq!(hits[0].rowid, 1, "{query:?}");
        assert_eq!(displayed_hits.len(), 1, "{query:?}");
        assert_eq!(displayed_hits[0].rowid, hits[0].rowid);
        assert_eq!(displayed_hits[0].score, hits[0].score);
    }
}

#[test]
fn how_query_alias_and_hex_text_are_queries() {
    let fixture = settings_fixture("unicode61");
    let output = run_lode(&fixture, "json", &["analyze", "how", "deadbeef"]);
    assert!(output.status.success());
    assert_eq!(json(&output)["query"], "deadbeef");
}

#[test]
fn how_query_human_and_table_views_escape_query_text() {
    let fixture = settings_fixture("simple");
    let compact = run_lode(&fixture, "compact", &["assay", "how", "a\nb"]);
    assert!(compact.status.success());
    let compact_stdout = String::from_utf8_lossy(&compact.stdout);
    assert!(compact_stdout.contains("a\\nb"));
    assert!(compact.stderr.is_empty());

    let extended = run_lode(&fixture, "extended", &["assay", "how", "a\nb"]);
    assert!(extended.status.success());
    assert!(extended.stderr.is_empty());

    let table = run_lode(&fixture, "table", &["assay", "how", "a\tb"]);
    assert!(table.status.success());
    let table_stdout = String::from_utf8_lossy(&table.stdout);
    let mut rows = table_stdout.lines();
    assert_eq!(
        rows.next(),
        Some("QUERY\tSTRATEGY\tTOKENIZER\tHELPER\tMATCH")
    );
    let row = rows.next().expect("query table row");
    assert_eq!(row.split('\t').count(), 5);
    assert!(row.starts_with("a\\tb\tsimple\tsimple\tsimple_query"));
}

#[test]
fn how_query_reports_empty_trigram_expression_and_invalid_query() {
    let fixture = settings_fixture("trigram");
    let short = run_lode(&fixture, "json", &["assay", "how", "ab"]);
    assert!(short.status.success());
    assert_eq!(json(&short)["match_expression"], "");

    let empty = run_lode(&fixture, "json", &["assay", "how", "  "]);
    assert_eq!(empty.status.code(), Some(1));
    assert_eq!(json(&empty)["code"], "invalid_query");
    assert!(empty.stderr.is_empty());
}

#[test]
fn how_query_does_not_scan_or_change_an_existing_index() {
    let fixture = fixture("unicode61", "indexed text");
    fs::write(
        fixture.workspace.join("notes.txt"),
        "changed workspace text",
    )
    .unwrap();

    let output = run_lode(&fixture, "json", &["assay", "how", "query"]);
    assert!(output.status.success());
    assert!(output.stderr.is_empty());

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
fn debug_tokens_human_views_show_terms_and_stale_warning() {
    let fixture = fixture("unicode61", "one one two");
    fs::write(
        fixture.workspace.join("notes.txt"),
        "changed workspace text",
    )
    .unwrap();

    let compact = run_lode(&fixture, "compact", &["debug", "tokens", &fixture.digest]);
    assert!(compact.status.success());
    let compact_stdout = String::from_utf8_lossy(&compact.stdout);
    assert!(compact_stdout.contains("Debug tokens"));
    assert!(compact_stdout.contains("Terms: 2 distinct · 3 occurrences"));
    assert!(compact_stdout.contains("one · two"));
    assert!(compact_stdout.contains("Notes\\twith newline\\n"));
    assert!(String::from_utf8_lossy(&compact.stderr).contains("stale index snapshot"));

    let extended = run_lode(&fixture, "extended", &["debug", "tokens", &fixture.digest]);
    assert!(extended.status.success());
    let extended_stdout = String::from_utf8_lossy(&extended.stdout);
    assert!(extended_stdout.contains("Text:"));
    assert!(extended_stdout.contains("one one two"));
    assert!(extended_stdout.contains("[0] one"));
    assert!(extended_stdout.contains("[2] two"));
    assert!(String::from_utf8_lossy(&extended.stderr).contains("stale index snapshot"));

    let table = run_lode(&fixture, "table", &["debug", "tokens", &fixture.digest]);
    assert!(table.status.success());
    let table_stdout = String::from_utf8_lossy(&table.stdout);
    let mut rows = table_stdout.lines();
    assert_eq!(
        rows.next(),
        Some("INDEX\tSURFACE\tVARIANTS\tDIGEST\tPATH\tHEADING\tPAGE\tSTATE")
    );
    let first = rows.next().expect("first token row");
    assert_eq!(first.split('\t').count(), 8);
    assert!(first.contains("\t[]\t"));
    assert!(first.contains("Notes\\twith newline\\n"));
    assert_eq!(rows.count(), 2);
    assert!(String::from_utf8_lossy(&table.stderr).contains("stale index snapshot"));

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
fn debug_tokens_compact_limits_distinct_terms() {
    let fixture = fixture(
        "unicode61",
        "one two three four five six seven eight nine ten eleven twelve thirteen",
    );
    let output = run_lode(&fixture, "compact", &["debug", "tokens", &fixture.digest]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Terms: 13 distinct · 13 occurrences"));
    assert!(stdout.contains("… 1 more term(s)"));
    assert!(!stdout.contains("thirteen"));
    assert!(output.stderr.is_empty());
}

#[test]
fn debug_tokens_extended_preserves_occurrences_and_variants() {
    let fixture = fixture("simple", "知识知识");
    let output = run_lode(&fixture, "extended", &["debug", "tokens", &fixture.digest]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Counts: 4 token occurrences · 2 distinct terms"));
    assert!(stdout.contains("[0] 知(z/zhi)"));
    assert!(stdout.contains("[2] 知(z/zhi)"));
    assert!(output.stderr.is_empty());
}

#[test]
fn debug_tokens_extended_lists_sources_and_visible_text_controls() {
    let fixture = fixture("unicode61", "line one\nline\t two\u{0001}");
    let second = fixture.workspace.join("other.txt");
    fs::write(&second, "line one\nline\t two\u{0001}").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert!(
        store
            .reference_file(&record_for(&second, &fixture.workspace), "text")
            .unwrap()
    );
    drop(store);

    let output = run_lode(&fixture, "extended", &["debug", "tokens", &fixture.digest]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("notes.txt (fresh)"));
    assert!(stdout.contains("other.txt (fresh)"));
    assert!(stdout.contains("line one\nline\t two\\u{1}"));
    assert!(output.stderr.is_empty());
}

#[cfg(unix)]
#[test]
fn debug_tokens_extended_escapes_control_characters_in_source_paths() {
    let fixture = fixture("unicode61", "path control");
    let second = fixture.workspace.join("other\nname.txt");
    fs::write(&second, "path control").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert!(
        store
            .reference_file(&record_for(&second, &fixture.workspace), "text")
            .unwrap()
    );
    drop(store);

    let output = run_lode(&fixture, "extended", &["debug", "tokens", &fixture.digest]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("other\\nname.txt (fresh)"));
    assert!(output.stderr.is_empty());
}

#[test]
fn debug_tokens_extended_escapes_control_characters_in_trigram_terms() {
    let fixture = fixture("trigram", "ab\ncd");
    let output = run_lode(&fixture, "extended", &["debug", "tokens", &fixture.digest]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[0] ab\\n"));
    assert!(stdout.contains("[1] b\\nc"));
    assert!(stdout.contains("[2] \\ncd"));
    assert!(output.stderr.is_empty());
}

#[test]
fn debug_tokens_empty_chunk_has_empty_human_and_table_results() {
    let fixture = fixture("unicode61", "");

    let compact = run_lode(&fixture, "compact", &["debug", "tokens", &fixture.digest]);
    assert!(compact.status.success());
    assert!(String::from_utf8_lossy(&compact.stdout).contains("(no indexed terms)"));

    let extended = run_lode(&fixture, "extended", &["debug", "tokens", &fixture.digest]);
    assert!(extended.status.success());
    assert!(String::from_utf8_lossy(&extended.stdout).contains("(no indexed terms)"));

    let table = run_lode(&fixture, "table", &["debug", "tokens", &fixture.digest]);
    assert!(table.status.success());
    assert_eq!(
        String::from_utf8_lossy(&table.stdout),
        "INDEX\tSURFACE\tVARIANTS\tDIGEST\tPATH\tHEADING\tPAGE\tSTATE\n"
    );
}

#[test]
fn debug_tokens_json_reports_stale_snapshot_without_reindexing() {
    let fixture = fixture("unicode61", "old indexed text");
    fs::write(fixture.workspace.join("notes.txt"), "new workspace text").unwrap();
    let output = run_lode(&fixture, "json", &["debug", "tokens", &fixture.digest]);
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
fn debug_tokens_json_reports_deleted_source_as_stale_snapshot() {
    let fixture = fixture("unicode61", "deleted source text");
    fs::remove_file(fixture.workspace.join("notes.txt")).unwrap();

    let output = run_lode(&fixture, "json", &["debug", "tokens", &fixture.digest]);
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
fn debug_tokens_json_reports_no_index_and_invalid_digest() {
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
    let no_index = run_lode(&missing_fixture, "json", &["debug", "tokens", "dead"]);
    assert_eq!(no_index.status.code(), Some(1));
    assert_eq!(json(&no_index)["code"], "no_index");

    let existing = fixture("unicode61", "hello");
    let invalid = run_lode(&existing, "json", &["debug", "tokens", "not-hex"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert_eq!(json(&invalid)["code"], "invalid_digest");
}

#[test]
fn debug_tokens_json_reports_ambiguous_prefix_without_chunk_text() {
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

    let output = run_lode(&fixture, "json", &["debug", "tokens", "dead"]);
    assert_eq!(output.status.code(), Some(1));
    let payload = json(&output);
    assert_eq!(payload["code"], "ambiguous");
    assert_eq!(payload["candidates"].as_array().unwrap().len(), 2);
    assert!(payload["candidates"][0].get("text").is_none());

    for view in ["compact", "extended", "table"] {
        let output = run_lode(&fixture, view, &["debug", "tokens", "dead"]);
        assert_eq!(output.status.code(), Some(1), "view={view}");
        assert!(output.stdout.is_empty(), "view={view}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("dead0001"), "view={view}: {stderr}");
        assert!(stderr.contains("notes.txt"), "view={view}: {stderr}");
        assert!(stderr.contains("dead0002"), "view={view}: {stderr}");
        assert!(stderr.contains("other.txt"), "view={view}: {stderr}");
    }
}

#[test]
fn debug_tokens_json_rejects_a_physical_tokenizer_mismatch() {
    let fixture = fixture("unicode61", "hello");
    fs::write(
        fixture.workspace.join("lode.toml"),
        "[fts]\nstrategy = \"trigram\"\n",
    )
    .unwrap();
    let output = run_lode(&fixture, "json", &["debug", "tokens", &fixture.digest]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(json(&output)["code"], "index_open_error");
}
