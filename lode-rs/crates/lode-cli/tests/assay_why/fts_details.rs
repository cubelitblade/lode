#![warn(clippy::pedantic)]

use super::*;

pub(super) fn lexical_config(strategy: &str) -> String {
    format!(
        "{}[retrieval.fusion.linear]\nsemantic_factor = 0.0\nlexical_factor = 1.0\n",
        config("http://127.0.0.1:1").replace("unicode61", strategy)
    )
}

fn success(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    json(output)
}

fn close(left: f64, right: f64) {
    assert!(
        (left - right).abs() <= 1e-12 + 1e-12 * right.abs(),
        "{left} != {right}"
    );
}

#[test]
fn requested_details_preserve_default_json_and_prospect_scores_for_aliases_and_digests() {
    let server = spawn_embedding_server(5);
    let fixture = fixture(&config(&server.endpoint), "alpha beta alpha");
    let default = success(&run_lode(
        &fixture,
        "json",
        &[
            "assay",
            "why",
            &fixture.digest,
            "alpha beta",
            "--top-k",
            "2",
        ],
    ));
    assert_eq!(default.as_object().unwrap().len(), 15);
    assert!(default.get("lexical_details").is_none());
    for (command, digest) in [
        ("assay", fixture.digest.as_str()),
        ("analyze", &fixture.digest[7..]),
        ("assay", "dead0001"),
    ] {
        let output = run_lode(
            &fixture,
            "json",
            &[
                command,
                "why",
                digest,
                "alpha beta",
                "--top-k",
                "2",
                "--fts-details",
            ],
        );
        let mut payload = success(&output);
        let details = payload
            .as_object_mut()
            .unwrap()
            .remove("lexical_details")
            .unwrap();
        assert_eq!(payload, default);
        assert_eq!(details["status"], "matched");
        assert_eq!(details["in_candidate_pool"], true);
        assert_eq!(details["match_expression"], "\"alpha\" OR \"beta\"");
        assert_eq!(
            details["tokenizer"],
            serde_json::json!({"strategy":"unicode61", "tokenize":"unicode61"})
        );
        assert_eq!(details["corpus_rows"], 1);
        assert_eq!(details["document_length"], 3);
        assert_eq!(details["average_length"], 3.0);
        assert_eq!(details.as_object().unwrap().len(), 9);
        let phrases = details["phrases"].as_array().unwrap();
        assert_eq!(phrases.len(), 2);
        close(
            phrases
                .iter()
                .map(|phrase| phrase["bm25_contribution"].as_f64().unwrap())
                .sum(),
            details["bm25"].as_f64().unwrap(),
        );
        close(
            details["bm25"].as_f64().unwrap(),
            payload["sources"]["lexical"]["raw_score"].as_f64().unwrap(),
        );
        assert_eq!(
            phrases[0]["hits"][0],
            serde_json::json!({"column":0, "token_offset":0, "document_tokens":["alpha"]})
        );
        assert_eq!(phrases[0]["hits"][1]["token_offset"], 2);
        assert!(details.get("rowid").is_none());
        assert!(details.get("contribution_sum").is_none());
    }
    let prospect = success(&run_lode(
        &fixture,
        "json",
        &["prospect", "alpha beta", "--top-k", "2"],
    ));
    close(
        prospect["hits"][0]["score"].as_f64().unwrap(),
        default["combined"].as_f64().unwrap(),
    );
    server.join();
}

#[test]
fn every_strategy_uses_real_expression_and_reports_physical_tokenizer() {
    for strategy in ["unicode61", "trigram", "simple", "jieba"] {
        let physical = if strategy == "jieba" {
            "simple"
        } else {
            strategy
        };
        let fixture = fixture_with_physical_tokenizer(
            &lexical_config(strategy),
            "知识 knowledge alpha beta",
            physical,
        );
        let details = success(&run_lode(
            &fixture,
            "json",
            &[
                "assay",
                "why",
                &fixture.digest,
                "知识 alpha",
                "--fts-details",
            ],
        ))["lexical_details"]
            .clone();
        let how = success(&run_lode(&fixture, "json", &["assay", "how", "知识 alpha"]));
        assert_eq!(details["status"], "matched", "{strategy}: {details}");
        assert_eq!(details["tokenizer"]["strategy"], strategy);
        assert_eq!(details["tokenizer"]["tokenize"], physical);
        assert_eq!(details["match_expression"], how["match_expression"]);
        assert!(!details["phrases"].as_array().unwrap().is_empty());
    }
}

#[test]
fn skipped_and_nonmatching_queries_have_null_statistics_and_no_phrases() {
    let server = spawn_embedding_server(1);
    let disabled = fixture(
        &format!(
            "{}[retrieval.fusion.linear]\nsemantic_factor = 1.0\nlexical_factor = 0.0\n",
            config(&server.endpoint)
        ),
        "alpha",
    );
    let empty =
        fixture_with_physical_tokenizer(&lexical_config("trigram"), "alpha beta", "trigram");
    let missing = fixture(&lexical_config("unicode61"), "alpha");
    for (fixture, query, status, expression) in [
        (&disabled, "alpha", "disabled", Value::Null),
        (&empty, "ab", "empty_query", Value::String(String::new())),
        (
            &missing,
            "quartz",
            "not_matched",
            Value::String("\"quartz\"".into()),
        ),
    ] {
        let payload = success(&run_lode(
            fixture,
            "json",
            &["assay", "why", &fixture.digest, query, "--fts-details"],
        ));
        let details = &payload["lexical_details"];
        assert_eq!(details["status"], status);
        assert_eq!(details["match_expression"], expression);
        assert_eq!(details["in_candidate_pool"], false);
        assert_eq!(details["phrases"], serde_json::json!([]));
        for field in ["bm25", "corpus_rows", "document_length", "average_length"] {
            assert!(details[field].is_null());
        }
    }
    server.join();
}

#[test]
fn matched_outside_candidate_pool_retains_absent_ranking_factors() {
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
    let pool = store
        .sparse_search(
            &lode_core::fts::match_query("alpha", "unicode61").unwrap(),
            4,
        )
        .unwrap();
    let rows = store
        .sparse_search(
            &lode_core::fts::match_query("alpha", "unicode61").unwrap(),
            100,
        )
        .unwrap();
    let rowid = rows
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
    let default = success(&run_lode(
        &fixture,
        "json",
        &["assay", "why", &target, "alpha", "--top-k", "1"],
    ));
    let mut payload = success(&run_lode(
        &fixture,
        "json",
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
    let details = payload
        .as_object_mut()
        .unwrap()
        .remove("lexical_details")
        .unwrap();
    assert_eq!(payload, default);
    assert_eq!(details["status"], "matched");
    assert_eq!(details["in_candidate_pool"], false);
    assert!(details["bm25"].is_number());
    assert!(payload["sources"]["lexical"]["raw_score"].is_null());
    assert!(payload["ranking_factors"].as_object().unwrap().is_empty());
    assert_eq!(payload["combined"], 0.0);
    assert!(payload["rank"].is_null());
    assert_eq!(payload["in_results"], false);
}

#[test]
fn stale_multiple_references_and_unrelated_changes_preserve_indexed_data() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha beta");
    let second = fixture.workspace.join("copy.txt");
    fs::write(&second, "alpha beta").unwrap();
    let unrelated = fixture.workspace.join("unrelated.txt");
    fs::write(&unrelated, "unrelated").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert!(
        store
            .reference_file(&record_for(&second, &fixture.workspace), "text")
            .unwrap()
    );
    store
        .replace_file(
            &record_for(&unrelated, &fixture.workspace),
            "text",
            &[Chunk {
                digest: format!("blake3:{:064x}", 42),
                text: "unrelated".into(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            Some(&[vec![0.0, 1.0, 0.0, 0.0]]),
        )
        .unwrap();
    let rows = store.find_chunk_rowids(&fixture.digest[7..]).unwrap();
    let before_text = store.get_chunks(&rows).unwrap()[&rows[0]].text.clone();
    let before_dense = store.dense_search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
    drop(store);
    let before = success(&run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    ));
    fs::write(&unrelated, "changed unrelated file").unwrap();
    let only_unrelated = success(&run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    ));
    assert_eq!(only_unrelated, before);
    fs::write(
        fixture.workspace.join("notes.txt"),
        "changed source with quartz",
    )
    .unwrap();
    let stale = success(&run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    ));
    assert_eq!(stale["has_stale"], true);
    assert_eq!(stale["paths"].as_array().unwrap().len(), 2);
    assert_eq!(stale["lexical_details"], before["lexical_details"]);
    assert_eq!(stale["sources"], before["sources"]);
    assert_eq!(stale["combined"], before["combined"]);
    let after = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert_eq!(after.get_chunks(&rows).unwrap()[&rows[0]].text, before_text);
    assert_eq!(
        after.dense_search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap(),
        before_dense
    );
}

#[test]
fn unsupported_views_fail_before_open_scan_or_embedding() {
    let fixture = fixture(&config("http://127.0.0.1:1"), "alpha");
    fs::write(fixture.workspace.join("notes.txt"), "changed").unwrap();
    for view in ["compact", "table"] {
        let output = run_lode(
            &fixture,
            view,
            &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
        );
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("--view json"));
    }
    let store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert_eq!(store.list_files().unwrap()[0].status, FileStatus::Fresh);
    drop(store);
    fs::remove_file(fixture.workspace.join(".lode/index.db")).unwrap();
    let output = run_lode(
        &fixture,
        "compact",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("--fts-details requires"));
}

#[test]
fn validation_and_lookup_errors_precede_stale_mutation_and_keep_json_contract() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha");
    let second = fixture.workspace.join("other.txt");
    fs::write(&second, "beta").unwrap();
    let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    store
        .replace_file(
            &record_for(&second, &fixture.workspace),
            "text",
            &[Chunk {
                digest: "blake3:dead000200000000000000000000000000000000000000000000000000000000"
                    .into(),
                text: "beta".into(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            Some(&[vec![1.0, 0.0, 0.0, 0.0]]),
        )
        .unwrap();
    drop(store);
    fs::write(fixture.workspace.join("notes.txt"), "changed source").unwrap();
    for (digest, query, args, code) in [
        ("not-hex", "alpha", vec![], "invalid_digest"),
        ("beef", "alpha", vec![], "not_found"),
        ("dead", "alpha", vec![], "ambiguous"),
        (fixture.digest.as_str(), "  ", vec![], "invalid_query"),
        (
            fixture.digest.as_str(),
            "alpha",
            vec!["--top-k", "0"],
            "invalid_top_k",
        ),
    ] {
        let mut command = vec!["assay", "why", digest, query, "--fts-details"];
        command.extend(args);
        let output = run_lode(&fixture, "json", &command);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stderr.is_empty());
        let error = json(&output);
        assert_eq!(error["code"], code);
        assert!(error["message"].is_string());
        assert!(error["hint"].is_string());
        assert!(error.get("lexical_details").is_none());
    }
    let store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
    assert!(
        store
            .list_files()
            .unwrap()
            .iter()
            .all(|file| file.status == FileStatus::Fresh)
    );
    drop(store);
    fs::remove_file(fixture.workspace.join(".lode/index.db")).unwrap();
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(json(&output)["code"], "no_index");
    assert!(output.stderr.is_empty());
}

#[test]
fn explicit_path_preserves_embedding_dimension_and_clap_errors() {
    let unavailable = fixture(&config("http://127.0.0.1:1"), "alpha");
    let error = run_lode(
        &unavailable,
        "json",
        &[
            "assay",
            "why",
            &unavailable.digest,
            "alpha",
            "--fts-details",
        ],
    );
    assert_eq!(error.status.code(), Some(1));
    assert_eq!(json(&error)["code"], "embedding_error");
    assert!(error.stderr.is_empty());
    let server = spawn_embedding_server_with_dimension(1, 3);
    let fixture = fixture(&config(&server.endpoint), "alpha");
    let error = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert_eq!(error.status.code(), Some(1));
    assert_eq!(json(&error)["code"], "dimension_error");
    assert!(error.stderr.is_empty());
    server.join();
    let error = run_lode(
        &fixture,
        "json",
        &["assay", "how", "alpha", "--fts-details"],
    );
    assert_eq!(error.status.code(), Some(2));
    assert!(error.stdout.is_empty());
    assert!(!error.stderr.is_empty());
    let help = run_lode(&fixture, "json", &["assay", "why", "--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("--fts-details"));
}

#[test]
fn diagnostic_failure_returns_one_json_error_and_no_partial_success() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha beta");
    let path = fixture.workspace.join(".lode/index.db");
    let connection = rusqlite::Connection::open(&path).unwrap();
    // An incompatible contentless table can still rank, but cannot provide
    // document sizes. Keep metadata intact to reach the diagnostic boundary.
    connection
        .execute_batch(
            "DROP TABLE chunks_fts;
         CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='', columnsize=0);
         INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks;",
        )
        .unwrap();
    drop(connection);
    let default = success(&run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha"],
    ));
    assert!(default["sources"]["lexical"]["raw_score"].is_number());
    let output = run_lode(
        &fixture,
        "json",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let payload = json(&output);
    assert_eq!(payload["code"], "fts_explanation_error");
    assert!(
        payload["message"]
            .as_str()
            .unwrap()
            .contains("negative FTS5 size")
    );
    assert!(payload["hint"].is_string());
    assert_eq!(payload.as_object().unwrap().len(), 3);
    assert!(payload.get("lexical_details").is_none());
    assert!(payload.get("combined").is_none());
}
