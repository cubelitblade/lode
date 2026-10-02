#![warn(clippy::pedantic)]

use super::fts_details::lexical_config;
use super::*;

fn human(output: &Output) -> String {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout.clone()).unwrap()
}

fn displayed_number(text: &str, label: &str) -> f64 {
    text.lines()
        .find_map(|line| line.trim().strip_prefix(label))
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn extended_flag_and_alias_keep_scores_and_fusion_with_one_embedding_per_call() {
    let server = spawn_embedding_server(4);
    let fixture = fixture(&config(&server.endpoint), "alpha beta alpha");
    let json_output = run_lode(
        &fixture,
        "json",
        &[
            "assay",
            "why",
            &fixture.digest,
            "alpha beta",
            "--fts-details",
        ],
    );
    assert!(json_output.status.success());
    let baseline = json(&json_output);
    for command in ["assay", "analyze"] {
        let text = human(&run_lode(
            &fixture,
            "extended",
            &[
                command,
                "why",
                &fixture.digest,
                "alpha beta",
                "--fts-details",
            ],
        ));
        assert!(text.contains("FTS details (raw BM25):"));
        assert!(text.contains("full MATCH status: matched"));
        assert!(text.contains("in lexical candidate pool: yes"));
        assert!(text.contains("MATCH: \"alpha\" OR \"beta\""));
        assert!(text.contains("phrase #0:"));
        assert!(text.contains("phrase #1:"));
        assert!(text.contains("hits=2"));
        assert!(text.contains("hits=1"));
        assert!(text.contains("Fusion explanation:"));
        assert!(text.contains("normalization:"));
        assert!(text.contains("Semantic similarity"));
        assert!(text.contains("Lexical relevance"));
        let combined = displayed_number(&text, "combined: ");
        assert!((combined - baseline["combined"].as_f64().unwrap()).abs() < 0.00005);
        let bm25 = displayed_number(&text, "raw BM25 (display rounded): ");
        assert!((bm25 - baseline["lexical_details"]["bm25"].as_f64().unwrap()).abs() < 1e-9);
        assert!(!text.contains("did not participate in fusion"));
    }
    let default = human(&run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "alpha beta"],
    ));
    assert!(!default.contains("FTS details"));
    assert!(!default.contains("phrase #"));
    assert!(default.contains("Fusion explanation:"));
    server.join();
}

#[test]
fn four_strategies_retain_duplicate_phrases_zero_branches_and_physical_tokenizer() {
    for strategy in ["unicode61", "trigram", "simple", "jieba"] {
        let physical = if strategy == "jieba" {
            "simple"
        } else {
            strategy
        };
        let fixture = fixture_with_physical_tokenizer(
            &lexical_config(strategy),
            "知识 alpha alpha",
            physical,
        );
        let query = if matches!(strategy, "simple" | "jieba") {
            "alpha alpha"
        } else {
            "alpha alpha zzz"
        };
        let output = run_lode(
            &fixture,
            "json",
            &["assay", "why", &fixture.digest, query, "--fts-details"],
        );
        assert!(output.status.success());
        let payload = json(&output);
        assert_eq!(payload["lexical_details"]["status"], "matched");
        let phrases = payload["lexical_details"]["phrases"].as_array().unwrap();
        let text = human(&run_lode(
            &fixture,
            "extended",
            &["assay", "why", &fixture.digest, query, "--fts-details"],
        ));
        assert!(text.contains(&format!("strategy={strategy} · physical={physical}")));
        assert_eq!(
            text.lines()
                .filter(|line| line.contains("phrase #"))
                .count(),
            phrases.len()
        );
        for (index, phrase) in phrases.iter().enumerate() {
            let line = text
                .lines()
                .find(|line| line.contains(&format!("phrase #{index}:")))
                .unwrap();
            assert!(line.contains(&format!(
                "hits={}",
                phrase["hits"].as_array().unwrap().len()
            )));
        }
        if phrases
            .iter()
            .any(|phrase| phrase["hits"].as_array().unwrap().is_empty())
        {
            assert!(
                text.lines()
                    .any(|line| line.contains("phrase #") && line.contains("hits=0"))
            );
        }
        if phrases.iter().any(|phrase| {
            phrase["hits"].as_array().unwrap().iter().any(|hit| {
                hit["document_tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(Value::is_null)
            })
        }) {
            assert!(text.contains("unknown document token"));
        }
        assert!(text.contains("signed raw BM25 only"));
        assert!(text.contains("including RRF"));
    }
}

#[test]
fn disabled_empty_and_no_match_explain_skipped_statistics() {
    let server = spawn_embedding_server(1);
    let disabled = fixture(
        &format!(
            "{}[retrieval.fusion.linear]\nsemantic_factor = 1.0\nlexical_factor = 0.0\n",
            config(&server.endpoint)
        ),
        "alpha",
    );
    let empty = fixture_with_physical_tokenizer(&lexical_config("trigram"), "alpha", "trigram");
    let missing = fixture(&lexical_config("unicode61"), "alpha");
    for (fixture, query, status, reason) in [
        (
            &disabled,
            "alpha",
            "disabled",
            "Lexical retrieval is disabled",
        ),
        (&empty, "ab", "empty_query", "lexical expression is empty"),
        (
            &missing,
            "quartz",
            "not_matched",
            "does not satisfy the full MATCH",
        ),
    ] {
        let text = human(&run_lode(
            fixture,
            "extended",
            &["assay", "why", &fixture.digest, query, "--fts-details"],
        ));
        assert!(text.contains(&format!("full MATCH status: {status}")));
        assert!(text.contains("in lexical candidate pool: no"));
        assert!(text.contains(reason));
        assert!(!text.contains("phrase #"));
        assert!(!text.contains("corpus rows:"));
        assert!(!text.contains("raw BM25 (display rounded):"));
    }
    server.join();
}

#[test]
fn pool_out_match_explicitly_reports_unused_bm25_and_keeps_no_rank() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha");
    fs::write(fixture.workspace.join("notes.txt"), "alpha\n".repeat(6)).unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    let chunks: Vec<_> = (0..6)
        .map(|seq| Chunk {
            digest: format!("blake3:{seq:064x}"),
            text: "alpha".into(),
            seq,
            heading: String::new(),
            page: None,
            span: None,
        })
        .collect();
    store
        .replace_file(
            &record_for(&fixture.workspace.join("notes.txt"), &fixture.workspace),
            "text",
            &chunks,
            Some(&vec![vec![1.0, 0.0, 0.0, 0.0]; 6]),
        )
        .unwrap();
    let expression = lode_core::fts::match_query("alpha", "unicode61").unwrap();
    let pool = store.sparse_search(&expression, 4).unwrap();
    let all = store.sparse_search(&expression, 100).unwrap();
    let rowid = all
        .iter()
        .find(|hit| !pool.iter().any(|candidate| candidate.rowid == hit.rowid))
        .unwrap()
        .rowid;
    let target = store
        .get_chunks(&[rowid])
        .unwrap()
        .remove(&rowid)
        .unwrap()
        .digest;
    drop(store);
    let text = human(&run_lode(
        &fixture,
        "extended",
        &[
            "assay",
            "why",
            &target,
            "alpha",
            "--top-k",
            "1",
            "--fts-details",
        ],
    ));
    assert!(text.contains("full MATCH status: matched"));
    assert!(text.contains("in lexical candidate pool: no"));
    assert!(text.contains("BM25 score did not participate in fusion"));
    assert!(text.contains("outside the lexical candidate pool"));
    assert!(text.contains("status: not retrieved"));
    assert!(text.contains("not ranked (no positive score)"));
    assert!(text.contains("in results: no"));
    assert!(text.contains("phrase #0:"));
}

#[test]
fn stale_extended_details_keep_old_matches_and_emit_one_existing_warning() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha beta");
    fs::write(fixture.workspace.join("notes.txt"), "quartz changed source").unwrap();
    let output = run_lode(
        &fixture,
        "extended",
        &["analyze", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stderr
            .matches("explaining a stale indexed snapshot")
            .count(),
        1
    );
    assert!(text.contains("(stale)"));
    assert!(text.contains("full MATCH status: matched"));
    assert!(text.contains("query tokens=[\"alpha\"]"));
    assert!(!text.contains("quartz"));
}

#[test]
fn trigram_control_characters_are_visible_and_cannot_inject_terminal_sequences() {
    let fixture =
        fixture_with_physical_tokenizer(&lexical_config("trigram"), "a\nb a\u{1b}b", "trigram");
    for query in ["a\nb", "a\u{1b}b"] {
        let text = human(&run_lode(
            &fixture,
            "extended",
            &["assay", "why", &fixture.digest, query, "--fts-details"],
        ));
        assert!(text.contains("full MATCH status: matched"));
        assert!(text.contains("phrase #0:"));
        assert!(
            !text
                .chars()
                .any(|character| character.is_control() && character != '\n')
        );
    }
}

#[test]
fn explicit_diagnostic_failure_has_empty_stdout_and_sourced_error_on_stderr() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha beta");
    let connection = rusqlite::Connection::open(fixture.workspace.join(".lode/index.db")).unwrap();
    connection
        .execute_batch(
            "DROP TABLE chunks_fts;
         CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='', columnsize=0);
         INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks;",
        )
        .unwrap();
    drop(connection);
    let output = run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("FTS explanation SQLite error"));
    assert!(stderr.contains("negative FTS5 size"));
    assert!(!stderr.contains("FTS details"));
    assert!(!stderr.contains("{\"code\""));
}

#[test]
fn rrf_explanation_keeps_positional_factors_and_separate_raw_bm25_details() {
    let server = spawn_embedding_server(2);
    let fixture = fixture(
        &format!(
            "{}[retrieval.fusion]\ntype = \"rrf\"\n[retrieval.fusion.rrf]\nk = 11\n",
            config(&server.endpoint)
        ),
        "alpha beta",
    );
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert!(output.status.success());
    let baseline = json(&output);
    let text = human(&run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    ));
    assert!(text.contains("normalization: none (skipped by RRF)"));
    assert!(text.contains("method: RRF (k=11)"));
    assert!(text.contains("factor: rank = 1"));
    assert!(
        (displayed_number(&text, "combined: ") - baseline["combined"].as_f64().unwrap()).abs()
            < 0.00005
    );
    assert!(text.contains("raw BM25 (display rounded):"));
    assert!(text.contains("do not attribute normalized or fused scores (including RRF)"));
    assert!(text.contains("full MATCH status: matched"));
    server.join();
}
