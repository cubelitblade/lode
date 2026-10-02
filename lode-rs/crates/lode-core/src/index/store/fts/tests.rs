use std::error::Error as _;

use tempfile::TempDir;

use super::*;
use crate::fts::explanation::{FtsMatchDetails, FtsPhrase};
use crate::fts::{explain_query, match_query};
use crate::index::records::{FileRecord, FileStatus};
use crate::ingestion::types::Chunk;
use crate::relpath::WorkspacePath;

const DOCUMENTS: &[&str] = &[
    "alpha beta alpha alphabet 知识 中国 common",
    "alpha gamma 知识库 知识 common",
    "beta gamma 中国人 common",
    "alpha beta gamma common",
    "knowledge mining document retrieval common",
    "中华人民共和国 知识 common",
    "重庆 重复重复 知识 common",
    "alphabet alphanumeric beta common",
    "café résumé common",
    "alpha alpha alpha common",
    "alpha far far far far beta common",
];

fn fixture(strategy: &str) -> (TempDir, Store) {
    let directory = tempfile::tempdir().unwrap();
    let mut store = Store::open(&directory.path().join("index.db"), "test", 4, strategy).unwrap();
    let record = FileRecord {
        path: WorkspacePath::from_posix("document.txt"),
        digest: "blake3:fts-details-file".to_string(),
        mtime: 1.0,
        size: 100,
        status: FileStatus::Fresh,
    };
    let chunks: Vec<_> = (0..31)
        .map(|seq| Chunk {
            digest: format!("blake3:fts-details-chunk-{seq}"),
            text: DOCUMENTS.get(seq as usize).map_or_else(
                || format!("filler{seq} unrelated common"),
                |text| (*text).to_string(),
            ),
            seq,
            heading: String::new(),
            page: None,
            span: None,
        })
        .collect();
    store.replace_file(&record, "text", &chunks, None).unwrap();
    (directory, store)
}

fn assert_close(left: f64, right: f64) {
    assert!(
        left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= 1e-12 + 1e-12 * right.abs(),
        "{left} != {right}",
    );
}

fn verify(store: &Store, expression: &MatchExpr) -> Vec<FtsMatchDetails> {
    let changes = store.conn.total_changes();
    let target = store.explain_fts(expression, 1).unwrap();
    if target.match_expression.is_empty() {
        assert!(target.matched.is_none());
        return Vec::new();
    }
    let hits = store.sparse_search(expression, 100).unwrap();
    let details: Vec<_> = hits
        .iter()
        .map(|hit| {
            let explanation = store.explain_fts(expression, hit.rowid).unwrap();
            let matched = explanation.matched.unwrap();
            assert_eq!(matched.rowid, hit.rowid);
            assert_eq!(matched.corpus_rows, 31);
            assert_close(matched.bm25, hit.score);
            assert_close(matched.contribution_sum, hit.score);
            for phrase in &matched.phrases {
                assert!((0..=31).contains(&phrase.document_frequency));
                for occurrence in &phrase.hits {
                    assert_eq!(occurrence.column, 0);
                    assert!(occurrence.token_offset >= 0);
                    assert_eq!(occurrence.document_tokens.len(), phrase.query_tokens.len());
                }
            }
            matched
        })
        .collect();
    assert!(
        store
            .explain_fts(expression, 999)
            .unwrap()
            .matched
            .is_none()
    );
    assert_eq!(store.conn.total_changes(), changes);
    details
}

#[test]
fn all_strategies_match_native_bm25_and_real_query_helpers() {
    crate::run_with_isolated_jieba_cache(
        "index::store::fts::tests::all_strategies_match_native_bm25_and_real_query_helpers",
        || {
            for strategy in ["unicode61", "trigram", "simple", "jieba"] {
                let (_directory, store) = fixture(strategy);
                for query in [
                    "alpha beta",
                    "知识",
                    "zhishi",
                    "中华人民共和国",
                    "alpha alpha",
                    "ALPHA",
                    "café",
                    "alpha \" beta",
                    "absentword",
                    "a",
                ] {
                    let expression = match_query(query, strategy).unwrap();
                    let expected = explain_query(query, strategy).unwrap().match_expression;
                    let explained = store.explain_fts(&expression, 1).unwrap();
                    assert_eq!(explained.match_expression, expected, "{strategy}/{query}");
                    verify(&store, &expression);
                }
                let first = if strategy == "trigram" {
                    "alp"
                } else {
                    "alpha"
                };
                let second = if strategy == "trigram" { "bet" } else { "beta" };
                let missing = if strategy == "trigram" {
                    "zzz"
                } else {
                    "absentword"
                };
                for query in [
                    format!("\"{first}\" OR \"{second}\""),
                    format!("\"{first}\" AND \"{second}\""),
                    "\"alpha beta\"".to_string(),
                    "alph*".to_string(),
                    format!("\"{first}\" OR \"{first}\""),
                    format!("\"{first}\" NOT \"gamma\""),
                    "NEAR(alpha beta, 10)".to_string(),
                    format!("\"{first}\" OR \"{missing}\""),
                    "\"common\"".to_string(),
                ] {
                    let explained = verify(&store, &MatchExpr::Prebuilt(query.clone()));
                    assert!(!explained.is_empty(), "{strategy}/{query}");
                }
                assert!(verify(&store, &MatchExpr::Prebuilt(format!("\"{missing}\""))).is_empty());
                verify_phrase_semantics(&store, strategy, first, missing);
            }
        },
    );
}

fn verify_phrase_semantics(store: &Store, strategy: &str, first: &str, missing: &str) {
    let repeated = store
        .explain_fts(
            &MatchExpr::Prebuilt(format!("\"{first}\" OR \"{first}\"")),
            1,
        )
        .unwrap()
        .matched
        .unwrap();
    assert_eq!(repeated.phrases.len(), 2);
    assert_eq!(
        repeated.phrases[0].query_tokens,
        repeated.phrases[1].query_tokens
    );
    assert_close(
        repeated.phrases[0].weighted_frequency,
        if strategy == "trigram" { 3.0 } else { 2.0 },
    );
    assert_close(
        repeated.phrases[0].bm25_contribution,
        repeated.phrases[1].bm25_contribution,
    );
    let unmatched = store
        .explain_fts(
            &MatchExpr::Prebuilt(format!("\"{first}\" OR \"{missing}\"")),
            1,
        )
        .unwrap()
        .matched
        .unwrap();
    assert_eq!(unmatched.phrases[1].document_frequency, 0);
    assert!(unmatched.phrases[1].hits.is_empty());
    assert_close(unmatched.phrases[1].bm25_contribution, 0.0);
    let prefix = store
        .explain_fts(&MatchExpr::Prebuilt("alph*".to_string()), 1)
        .unwrap()
        .matched
        .unwrap();
    assert!(
        prefix.phrases[0]
            .hits
            .iter()
            .all(|hit| hit.document_tokens.last() == Some(&None))
    );
    let common = store
        .explain_fts(&MatchExpr::Prebuilt("\"common\"".to_string()), 1)
        .unwrap()
        .matched
        .unwrap();
    assert_eq!(common.phrases[0].document_frequency, 31);
    assert_close(common.phrases[0].idf, 1e-6);
    if matches!(strategy, "simple" | "jieba") {
        for query in ["\"zhi\" AND \"shi\"", "\"知\" OR \"zhi\"", "\"知识\""] {
            assert!(!verify(store, &MatchExpr::Prebuilt(query.to_string())).is_empty());
        }
        let colocated = store
            .explain_fts(&MatchExpr::Prebuilt("\"知\" OR \"zhi\"".to_string()), 1)
            .unwrap()
            .matched
            .unwrap();
        assert_eq!(
            colocated.phrases[0].hits[0].token_offset,
            colocated.phrases[1].hits[0].token_offset
        );
    }
}

#[test]
fn target_filter_keeps_full_corpus_statistics_and_does_not_depend_on_pool() {
    let (_directory, store) = fixture("unicode61");
    let expression = MatchExpr::Prebuilt("\"alpha\"".to_string());
    let pool = store.sparse_search(&expression, 1).unwrap();
    let all = store.sparse_search(&expression, 100).unwrap();
    let target = all
        .iter()
        .find(|hit| hit.rowid != pool[0].rowid)
        .unwrap()
        .rowid;
    let filtered = store
        .explain_fts(&expression, target)
        .unwrap()
        .matched
        .unwrap();
    let (bm25, json): (f64, String) = store
        .conn
        .query_row(
            "SELECT bm25(chunks_fts), lode_fts_details(chunks_fts)
         FROM chunks_fts WHERE chunks_fts MATCH ?1",
            ["\"alpha\""],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    let whole: RowDetails = serde_json::from_str(&json).unwrap();
    let rowid = whole.rowid;
    let whole = whole.into_match(rowid, bm25).unwrap();
    assert_eq!(filtered.corpus_rows, whole.corpus_rows);
    assert_close(filtered.average_length, whole.average_length);
    assert_eq!(
        filtered.phrases[0].document_frequency,
        whole.phrases[0].document_frequency
    );
    let native = all.iter().find(|hit| hit.rowid == target).unwrap().score;
    assert_close(filtered.bm25, native);
    assert!(
        store
            .explain_fts(
                &MatchExpr::Prebuilt("\"alpha\" AND \"absentword\"".to_string()),
                target,
            )
            .unwrap()
            .matched
            .is_none()
    );
}

#[test]
fn empty_expression_does_not_register_or_submit_match() {
    let (_directory, store) = fixture("unicode61");
    let result = store
        .explain_fts(&MatchExpr::Prebuilt(String::new()), 1)
        .unwrap();
    assert!(result.match_expression.is_empty());
    assert!(result.matched.is_none());
    assert!(store.conn.prepare(
        "SELECT lode_fts_details(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH 'alpha'",
    ).is_err());
}

#[test]
fn diagnostic_sqlite_and_helper_errors_preserve_causes_and_allow_retry() {
    let (_directory, store) = fixture("unicode61");
    let malformed = MatchExpr::Prebuilt("\"unterminated".to_string());
    let error = store.explain_fts(&malformed, 1).unwrap_err();
    let crate::Error::FtsExplanation(source) = &error else {
        panic!("{error:?}")
    };
    assert!(
        source
            .source()
            .unwrap()
            .downcast_ref::<rusqlite::Error>()
            .is_some()
    );
    assert!(error.to_string().contains("unterminated string"));
    assert!(
        store
            .explain_fts(
                &MatchExpr::Helper {
                    function: "unexpected_helper",
                    query: "alpha".to_string()
                },
                1,
            )
            .is_err()
    );
    let missing = store
        .explain_fts(
            &MatchExpr::Helper {
                function: "simple_query",
                query: "alpha".to_string(),
            },
            1,
        )
        .unwrap_err();
    assert!(missing.to_string().contains("no such function"));
    let crate::Error::FtsExplanation(source) = missing else {
        panic!("wrong error")
    };
    assert!(source.source().is_some());
    assert!(
        store
            .explain_fts(&match_query("alpha", "unicode61").unwrap(), 1)
            .unwrap()
            .matched
            .is_some()
    );
}

#[test]
fn reopened_indexes_support_opt_in_diagnostics() {
    let (directory, store) = fixture("simple");
    drop(store);
    let store = Store::open_existing(&directory.path().join("index.db")).unwrap();
    assert!(
        store
            .explain_fts(&match_query("知识", "simple").unwrap(), 1)
            .unwrap()
            .matched
            .is_some()
    );
}

#[test]
fn invalid_transport_and_score_disagreement_are_errors() {
    let (_directory, store) = fixture("unicode61");
    let matched = store
        .explain_fts(&match_query("alpha", "unicode61").unwrap(), 1)
        .unwrap()
        .matched
        .unwrap();
    let details = RowDetails {
        rowid: matched.rowid,
        corpus_rows: matched.corpus_rows,
        average_length: matched.average_length,
        document_length: matched.document_length,
        phrases: matched.phrases.clone(),
    };
    assert!(matches!(
        details.into_match(999, matched.bm25),
        Err(FtsExplanationError::InvalidData("target rowid mismatch"))
    ));
    let details = RowDetails {
        rowid: matched.rowid,
        corpus_rows: matched.corpus_rows,
        average_length: matched.average_length,
        document_length: matched.document_length,
        phrases: Vec::<FtsPhrase>::new(),
    };
    assert!(details.into_match(1, matched.bm25).is_err());
    assert!(serde_json::from_str::<RowDetails>("{}").is_err());
}

#[test]
fn empty_indexes_and_native_empty_helpers_return_no_match() {
    for strategy in ["unicode61", "simple"] {
        let directory = tempfile::tempdir().unwrap();
        let store = Store::open(&directory.path().join("index.db"), "test", 4, strategy).unwrap();
        let expression = match_query("alpha", strategy).unwrap();
        let result = store.explain_fts(&expression, 1).unwrap();
        assert!(!result.match_expression.is_empty());
        assert!(result.matched.is_none());
        if strategy == "simple" {
            let result = store
                .explain_fts(&match_query("", strategy).unwrap(), 1)
                .unwrap();
            assert!(result.match_expression.is_empty());
            assert!(result.matched.is_none());
        }
    }
}
