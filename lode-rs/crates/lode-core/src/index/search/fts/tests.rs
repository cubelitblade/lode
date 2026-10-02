use std::cell::RefCell;

use super::*;
use crate::embeddings::testing::{FakeEmbedder, file_record, make_chunks};
use crate::index::ranking::{Fusion, Norm};

fn plan(semantic: f64, lexical: f64) -> RetrievalPlan {
    RetrievalPlan {
        norm: Some(Norm::MinMax),
        fusion: Fusion::Linear {
            weights: [(Source::Semantic, semantic), (Source::Lexical, lexical)].into(),
        },
    }
}

fn fixture(strategy: &str, texts: &[&str]) -> (tempfile::TempDir, Store) {
    let directory = tempfile::tempdir().unwrap();
    let mut store = Store::open(
        &directory.path().join("index.db"),
        "test-model",
        4,
        strategy,
    )
    .unwrap();
    let (chunks, vectors) = make_chunks(texts, None);
    store
        .replace_file(
            &file_record("notes.txt", "blake3:notes", 1),
            "text",
            &chunks,
            Some(&vectors),
        )
        .unwrap();
    (directory, store)
}

#[test]
fn details_preserve_scores_and_embedding_count() {
    let (_dir, store) = fixture("unicode61", &["alpha beta", "alpha alpha", "unrelated"]);
    let embedder = FakeEmbedder::default();
    for plan in [
        plan(0.6, 0.4),
        plan(0.0, 1.0),
        RetrievalPlan {
            norm: None,
            fusion: Fusion::Rrf { k: 11 },
        },
    ] {
        let before = embedder.query_calls.get();
        let default = super::super::explain(&store, &embedder, "alpha beta", 1, &plan, 1).unwrap();
        let with = explain_with_fts(&store, &embedder, "alpha beta", 1, &plan, 1).unwrap();
        assert_eq!(default, with.explanation);
        assert_eq!(with.lexical_details.status, LexicalDetailsStatus::Matched);
        assert!(with.lexical_details.in_candidate_pool);
        let per_call = usize::from(source_enabled(&plan, Source::Semantic));
        assert_eq!(embedder.query_calls.get() - before, 2 * per_call);
        let matched = with.lexical_details.matched.unwrap();
        assert_eq!(matched.corpus_rows, 3);
        assert_eq!(matched.phrases.len(), 2);
        validate_pool_score(default.sources.get(&Source::Lexical), Some(&matched)).unwrap();
        let mut inconsistent = matched;
        inconsistent.bm25 += 1.0;
        assert!(
            validate_pool_score(default.sources.get(&Source::Lexical), Some(&inconsistent))
                .is_err()
        );
        assert!(validate_pool_score(default.sources.get(&Source::Lexical), None).is_err());
    }
}

#[test]
fn disabled_empty_and_nonmatching_are_successful_without_extra_embedding() {
    let (_dir, store) = fixture("trigram", &["alpha beta", "unrelated"]);
    let embedder = FakeEmbedder::default();
    for (query, plan, status, expression) in [
        (
            "alpha",
            plan(1.0, 0.0),
            LexicalDetailsStatus::Disabled,
            None,
        ),
        (
            "ab",
            plan(0.0, 1.0),
            LexicalDetailsStatus::EmptyQuery,
            Some(""),
        ),
        (
            "zzzz",
            plan(0.0, 1.0),
            LexicalDetailsStatus::NotMatched,
            Some("\"zzz\" OR \"zzz\""),
        ),
    ] {
        let result = explain_with_fts(&store, &embedder, query, 1, &plan, 1).unwrap();
        let details = result.lexical_details;
        assert_eq!(details.status, status);
        assert_eq!(details.match_expression.as_deref(), expression);
        assert!(!details.in_candidate_pool);
        assert!(details.matched.is_none());
    }
    assert_eq!(embedder.query_calls.get(), 1);
}

#[test]
fn matched_target_outside_pool_keeps_ranking_and_raw_score_absent() {
    let texts = ["alpha"; 6];
    let (_dir, store) = fixture("unicode61", &texts);
    let expression = crate::fts::match_query("alpha", "unicode61").unwrap();
    let pool = store.sparse_search(&expression, 4).unwrap();
    let rowid = (1..=6)
        .find(|id| !pool.iter().any(|hit| hit.rowid == *id))
        .unwrap();
    let plan = plan(0.0, 1.0);
    let embedder = FakeEmbedder::default();
    let default = super::super::explain(&store, &embedder, "alpha", rowid, &plan, 1).unwrap();
    let with = explain_with_fts(&store, &embedder, "alpha", rowid, &plan, 1).unwrap();
    assert_eq!(with.explanation, default);
    assert_eq!(with.lexical_details.status, LexicalDetailsStatus::Matched);
    assert!(!with.lexical_details.in_candidate_pool);
    assert!(with.lexical_details.matched.is_some());
    assert_eq!(with.explanation.sources[&Source::Lexical].raw_score, None);
    assert_eq!(with.explanation.rank, None);
    assert_eq!(embedder.query_calls.get(), 0);
}

struct DuringEmbedding {
    inner: FakeEmbedder,
    write: RefCell<Option<Box<dyn FnOnce()>>>,
}

impl Embedder for DuringEmbedding {
    fn model_id(&self) -> crate::Result<String> {
        self.inner.model_id()
    }
    fn dimension(&self) -> crate::Result<usize> {
        Ok(self.inner.dimension())
    }
    fn embed_documents(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(texts)
    }
    fn embed_query(&self, query: &str) -> crate::Result<Vec<f32>> {
        if let Some(write) = self.write.borrow_mut().take() {
            write();
        }
        self.inner.embed_query(query)
    }
}

#[test]
fn wal_writer_during_embedding_cannot_mix_target_candidates_or_statistics() {
    let (dir, store) = fixture("unicode61", &["alpha beta", "alpha"]);
    let plan = plan(0.6, 0.4);
    let baseline =
        explain_with_fts(&store, &FakeEmbedder::default(), "alpha", 1, &plan, 2).unwrap();
    assert!(baseline.lexical_details.matched.as_ref().unwrap().phrases.iter()
        .flat_map(|phrase| &phrase.hits).all(|hit| matches!(
            &hit.mapping, crate::fts::explanation::FtsHitMapping::Mapped(span) if span.surface == "alpha"
        )));
    let path = dir.path().join("index.db");
    let embedder = DuringEmbedding {
        inner: FakeEmbedder::default(),
        write: RefCell::new(Some(Box::new(move || {
            let mut writer = Store::open_existing(&path).unwrap();
            let (chunks, vectors) = make_chunks(&["new unrelated document"], None);
            writer
                .replace_file(
                    &file_record("notes.txt", "blake3:updated", 2),
                    "text",
                    &chunks,
                    Some(&vectors),
                )
                .unwrap();
        }))),
    };
    let result = explain_with_fts(&store, &embedder, "alpha", 1, &plan, 2).unwrap();
    assert_eq!(result, baseline);
    assert_eq!(embedder.inner.query_calls.get(), 1);
    let updated_rowids = store
        .find_chunk_rowids(&crate::ingestion::digest::chunk_digest("new unrelated document")[7..])
        .unwrap();
    assert_eq!(updated_rowids.len(), 1);
    assert_eq!(
        store.get_chunks(&updated_rowids).unwrap()[&updated_rowids[0]].text,
        "new unrelated document"
    );
    assert!(store.get_chunks(&[1]).unwrap().is_empty());
}

#[test]
fn scoring_failure_releases_snapshot_and_preserves_error() {
    let (dir, store) = fixture("unicode61", &["alpha"]);
    let embedder = FakeEmbedder::new("test-model", 8);
    let error = explain_with_fts(&store, &embedder, "alpha", 1, &plan(1.0, 0.0), 1).unwrap_err();
    assert!(matches!(
        error,
        crate::Error::DimensionMismatch {
            stored: 4,
            current: 8
        }
    ));
    let mut writer = Store::open_existing(&dir.path().join("index.db")).unwrap();
    writer
        .mark_stale(&crate::relpath::WorkspacePath::from_posix("notes.txt"))
        .unwrap();
    let result = explain_with_fts(
        &store,
        &FakeEmbedder::default(),
        "alpha",
        1,
        &plan(0.0, 1.0),
        1,
    )
    .unwrap();
    assert_eq!(
        result.explanation.chunk.refs[0].status,
        crate::index::records::FileStatus::Stale
    );
}
