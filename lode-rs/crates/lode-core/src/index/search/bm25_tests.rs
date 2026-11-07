#![warn(clippy::pedantic)]

use super::*;
use crate::embeddings::testing::{FakeEmbedder, file_record, make_chunks};
use crate::index::ranking::Norm;

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

fn lexical_plan(norm: Option<Norm>) -> RetrievalPlan {
    RetrievalPlan {
        norm,
        fusion: Fusion::Linear {
            weights: [(Source::Semantic, 0.0), (Source::Lexical, 1.0)].into(),
        },
    }
}

#[test]
fn bm25_candidates_keep_strong_matches_before_truncation_for_every_strategy() {
    let weak = format!("alpha {}", "filler ".repeat(100));
    let strong = "alpha alpha alpha alpha";
    for strategy in ["unicode61", "trigram", "simple", "jieba"] {
        let texts = [
            weak.as_str(),
            weak.as_str(),
            weak.as_str(),
            weak.as_str(),
            weak.as_str(),
            strong,
            "unrelated",
        ];
        let (_directory, store) = fixture(strategy, &texts);
        let expression = crate::fts::match_query("alpha", strategy).unwrap();
        let pool = store.sparse_search(&expression, 4).unwrap();
        assert_eq!(pool.len(), 4);
        assert_eq!(pool[0].rowid, 6, "{strategy}: {pool:?}");
        assert!(pool.iter().all(|hit| hit.score < 0.0));
        assert!(pool.windows(2).all(|pair| pair[0].score <= pair[1].score));
        assert_eq!(
            pool.iter().skip(1).map(|hit| hit.rowid).collect::<Vec<_>>(),
            [1, 2, 3]
        );
    }
}

#[test]
fn bm25_linear_search_and_explanations_reward_stronger_matches_for_every_norm() {
    let weak = format!("alpha {}", "filler ".repeat(100));
    let (_directory, store) = fixture(
        "unicode61",
        &[weak.as_str(), "alpha alpha alpha alpha", "unrelated"],
    );
    let embedder = FakeEmbedder::default();
    for norm in [
        None,
        Some(Norm::MinMax),
        Some(Norm::Softmax { temperature: 1.0 }),
    ] {
        let plan = lexical_plan(norm);
        let hits = search(&store, &embedder, "alpha", &plan, 2).unwrap();
        assert_eq!(hits[0].text, "alpha alpha alpha alpha", "{norm:?}");
        assert!(hits[0].score > 0.0);
        let weak = explain(&store, &embedder, "alpha", 1, &plan, 2).unwrap();
        let strong = explain(&store, &embedder, "alpha", 2, &plan, 2).unwrap();
        assert!(strong.combined > weak.combined, "{norm:?}");
        assert_eq!(strong.sources[&Source::Lexical].pool_rank, Some(1));
        let lexical = strong.sources[&Source::Lexical];
        assert!(lexical.raw_score.unwrap() < 0.0);
        assert!(lexical.prepared_score.unwrap() < 0.0);
        assert!((strong.combined + lexical.prepared_score.unwrap()).abs() < 1e-12);
        let diagnostic = explain_with_fts(&store, &embedder, "alpha", 2, &plan, 2).unwrap();
        assert_eq!(diagnostic.explanation, strong);
        assert!(diagnostic.lexical_details.in_candidate_pool);
        let details = diagnostic.lexical_details.matched.unwrap();
        assert_eq!(lexical.raw_score, Some(details.bm25));
        let sum: f64 = details
            .phrases
            .iter()
            .map(|phrase| phrase.bm25_contribution)
            .sum();
        assert!((sum - details.bm25).abs() < 1e-12);
        assert!((hits[0].score - strong.combined).abs() < 1e-12);
        let fusion = plan
            .fusion
            .explain(&strong.sources, strong.combined, plan.norm);
        assert!((fusion.ranking_factors[&Source::Lexical].value - strong.combined).abs() < 1e-12);
    }
    assert_eq!(embedder.query_calls.get(), 0);
}

#[test]
fn bm25_rrf_and_pool_ranks_use_ascending_scores_and_rowid_ties() {
    let weak = format!("alpha {}", "filler ".repeat(100));
    let (_directory, store) = fixture(
        "unicode61",
        &[
            weak.as_str(),
            "alpha alpha alpha alpha",
            "alpha alpha alpha alpha",
        ],
    );
    let embedder = FakeEmbedder::default();
    let plan = RetrievalPlan {
        norm: None,
        fusion: Fusion::Rrf { k: 10 },
    };
    for (rowid, lexical_rank) in [(1, 3), (2, 1), (3, 2)] {
        let result = explain_with_fts(&store, &embedder, "alpha", rowid, &plan, 3).unwrap();
        let explanation = result.explanation;
        let lexical = explanation.sources[&Source::Lexical];
        assert_eq!(lexical.pool_rank, Some(lexical_rank));
        assert_eq!(lexical.raw_score, lexical.prepared_score);
        assert!(lexical.raw_score.unwrap() < 0.0);
        let semantic_rank = explanation.sources[&Source::Semantic].pool_rank.unwrap();
        let expected = 1.0 / (10.0 + f64::from(u32::try_from(semantic_rank).unwrap()))
            + 1.0 / (10.0 + f64::from(u32::try_from(lexical_rank).unwrap()));
        assert!((explanation.combined - expected).abs() < 1e-12);
        let fusion = plan
            .fusion
            .clone()
            .explain(&explanation.sources, explanation.combined, None);
        assert!(
            (fusion.ranking_factors[&Source::Lexical].value
                - f64::from(u32::try_from(lexical_rank).unwrap()))
            .abs()
                < 1e-12
        );
    }
}
