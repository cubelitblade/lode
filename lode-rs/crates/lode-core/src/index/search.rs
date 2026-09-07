#![warn(clippy::pedantic)]

//! Hybrid retrieval: semantic (dense vec0) + lexical (sparse FTS5) fusion.
//!
//! Both sources are scored, then combined by a pluggable [`RetrievalPlan`]
//! (from `super::ranking`): a per-source [`Norm`] (min-max, softmax) followed
//! by a cross-source [`Fusion`] (weighted linear sum, reciprocal rank
//! fusion). Semantic scores are cosine similarities (L2-normalized vectors,
//! so cosine == dot); lexical scores are BM25.
//!
//! Mirrors `src/lode/index/search.py`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::embeddings::base::Embedder;
use crate::index::explanation::{RetrievalStatus, ScoreExplanation, SourceExplanation};
use crate::index::ranking::{Fusion, PreparedScores, RetrievalPlan};
use crate::index::records::{DenseMatch, FileStatus, PathRef, Source, SparseMatch};

use super::store::Store;

/// Retrieve a larger candidate pool than `top_k` from each source so the fused
/// ranking can still reach the best combined result.
const CANDIDATE_MULTIPLIER: usize = 4;

/// One fused retrieval result with its provenance.
///
/// Content is shared by identical files, so a hit carries every path
/// referencing it; `primary()` picks the representative one and `stale()`
/// reports whether any referencing path is outdated.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// Content digest: `blake3:<hex>`.
    pub digest: String,
    /// Chunk body text.
    pub text: String,
    /// Heading chain the chunk belongs to.
    pub heading: String,
    /// Combined score after fusion.
    pub score: f64,
    /// Every path referencing the chunk's content, with freshness.
    pub refs: Vec<PathRef>,
    /// Page number for PDF chunks.
    pub page: Option<u32>,
}

impl SearchHit {
    /// Representative reference: smallest fresh path, else smallest overall.
    ///
    /// # Panics
    ///
    /// Never: every hit is assembled from a join-derived chunk with at least
    /// one reference.
    #[must_use]
    pub fn primary(&self) -> PathRef {
        let mut fresh: Vec<&PathRef> = self
            .refs
            .iter()
            .filter(|r| r.status == FileStatus::Fresh)
            .collect();
        if fresh.is_empty() {
            fresh = self.refs.iter().collect();
        }
        // Safe: a chunk fetched through the store join always has refs.
        fresh
            .into_iter()
            .min_by(|a, b| a.path.as_str().cmp(b.path.as_str()))
            .expect("refs is never empty for a join-derived hit")
            .clone()
    }

    /// Whether any referencing path is stale.
    #[must_use]
    pub fn stale(&self) -> bool {
        self.refs.iter().any(|r| r.status == FileStatus::Stale)
    }
}

/// Aggregated output of a prospect command.
///
/// Carries the query context, the hits, and the library-wide dirty signal
/// (`has_stale`, from change detection), distinct from per-hit
/// [`SearchHit::stale`].
#[derive(Debug, Clone)]
pub struct ProspectResult {
    /// Workspace the query ran against.
    pub workspace: PathBuf,
    /// Raw query string.
    pub query: String,
    /// Requested result cap.
    pub top_k: u32,
    /// Fused hits in rank order.
    pub hits: Vec<SearchHit>,
    /// Whether any changed or missing file was detected before searching.
    pub has_stale: bool,
}

/// Intermediate per-source scores for one query, before ranking.
///
/// `raw` are the un-normalized per-source scores (cosine for semantic, BM25
/// for lexical); `prepared` are their values after the plan's norm (or the
/// raw values when the plan skips normalization, e.g. RRF); `combined` is
/// the fusion output. `search` ranks from `combined`; `explain` reads the
/// per-source values for a single chunk.
struct Candidates {
    semantic_raw: BTreeMap<i64, f64>,
    lexical_raw: BTreeMap<i64, f64>,
    prepared: PreparedScores,
    combined: BTreeMap<i64, f64>,
}

/// Hybrid search: fusion of semantic and lexical results per `plan`.
///
/// # Errors
///
/// Fails when the embedder cannot embed the query, a linear plan has both
/// factors zero, or the store queries fail.
pub fn search(
    store: &Store,
    embedder: &dyn Embedder,
    query: &str,
    plan: &RetrievalPlan,
    top_k: u32,
) -> crate::Result<Vec<SearchHit>> {
    let candidates = score_candidates(store, embedder, query, plan, top_k)?;
    if top_k == 0 || query.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Zero-score rows (no match in either source) carry no signal; dropping
    // them keeps e.g. sparse-only queries from returning unrelated chunks.
    // `sort_by` is stable, matching Python's sorted(reverse=True) tie order.
    let mut ranked: Vec<(i64, f64)> = candidates
        .combined
        .iter()
        .filter(|(_, score)| **score > 0.0)
        .map(|(&rowid, &score)| (rowid, score))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(top_k as usize);
    if ranked.is_empty() {
        return Ok(Vec::new());
    }

    let chunks = store.get_chunks(&ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>())?;
    let mut hits = Vec::new();
    for (rowid, score) in ranked {
        if let Some(chunk) = chunks.get(&rowid) {
            hits.push(SearchHit {
                digest: chunk.digest.clone(),
                text: chunk.text.clone(),
                heading: chunk.heading.clone(),
                score,
                refs: chunk.refs.clone(),
                page: chunk.page,
            });
        }
    }
    Ok(hits)
}

/// Score every candidate chunk for a query, before ranking.
///
/// Shared by `search` (which ranks from `combined`) and `explain` (which
/// reads the per-source values for one chunk), so the two never drift apart.
/// An empty query or non-positive `top_k` yields empty candidates; a linear
/// plan with both factors zero raises.
///
/// # Errors
///
/// Fails when the embedder or the store queries fail; also fails on a linear
/// plan whose factors are both zero.
fn score_candidates(
    store: &Store,
    embedder: &dyn Embedder,
    query: &str,
    plan: &RetrievalPlan,
    top_k: u32,
) -> crate::Result<Candidates> {
    if top_k == 0 || query.trim().is_empty() {
        return Ok(Candidates {
            semantic_raw: BTreeMap::new(),
            lexical_raw: BTreeMap::new(),
            prepared: BTreeMap::new(),
            combined: BTreeMap::new(),
        });
    }
    let both_zero = match &plan.fusion {
        Fusion::Linear { weights } => {
            weights.get(&Source::Semantic).copied().unwrap_or(0.0) == 0.0
                && weights.get(&Source::Lexical).copied().unwrap_or(0.0) == 0.0
        }
        Fusion::Rrf { .. } => false,
    };
    if both_zero {
        return Err(crate::Error::Config(
            "You can't discover an ore without a prospecting tool.\n\
             Hint: at least one of `semantic_factor` and `lexical_factor` must be non-zero."
                .to_string(),
        ));
    }
    let pool = u32::try_from(top_k as usize * CANDIDATE_MULTIPLIER).unwrap_or(u32::MAX);

    let semantic_raw = if source_enabled(plan, Source::Semantic) {
        let vector = embedder.embed_query(query)?;
        dense_to_cosine(store.dense_search(&vector, pool)?)
    } else {
        BTreeMap::new()
    };
    let lexical_raw = if source_enabled(plan, Source::Lexical) {
        let expr = crate::fts::match_query(query, &store.meta().tokenizer)?;
        sparse_to_map(store.sparse_search(&expr, pool)?)
    } else {
        BTreeMap::new()
    };

    let mut raw = PreparedScores::new();
    raw.insert(Source::Semantic, semantic_raw.clone());
    raw.insert(Source::Lexical, lexical_raw.clone());
    let prepared = prepare(plan, raw);
    let combined = plan.fusion.clone().fuse(&prepared);
    Ok(Candidates {
        semantic_raw,
        lexical_raw,
        prepared,
        combined,
    })
}

/// Whether a source is queried at all.
///
/// A zero linear factor disables the source; RRF always queries both.
fn source_enabled(plan: &RetrievalPlan, source: Source) -> bool {
    match &plan.fusion {
        Fusion::Linear { weights } => weights.get(&source).copied().unwrap_or(0.0) != 0.0,
        Fusion::Rrf { .. } => true,
    }
}

/// Apply the plan's norm per source, or pass raw through when the norm is
/// skipped.
fn prepare(plan: &RetrievalPlan, raw: PreparedScores) -> PreparedScores {
    let Some(norm) = plan.norm else {
        return raw;
    };
    raw.into_iter()
        .map(|(source, scores)| {
            let normalized = norm.normalize(&scores);
            (source, normalized)
        })
        .collect()
}

/// Cosine similarity from L2 distance over L2-normalized vectors.
///
/// For normalized vectors, d^2 = 2 - 2*cos, so cos = 1 - d^2/2.
fn cosine(distance: f64) -> f64 {
    1.0 - (distance * distance) / 2.0
}

/// Convert kNN matches to a rowid → cosine map.
fn dense_to_cosine(matches: Vec<DenseMatch>) -> BTreeMap<i64, f64> {
    matches
        .into_iter()
        .map(|m| (m.rowid, cosine(m.distance)))
        .collect()
}

/// Convert BM25 matches to a rowid → score map.
fn sparse_to_map(matches: Vec<SparseMatch>) -> BTreeMap<i64, f64> {
    matches.into_iter().map(|m| (m.rowid, m.score)).collect()
}

/// Explain why one chunk scored as it did for a query.
///
/// Reuses the same candidate scoring as `search` so the explanation can
/// never drift from the ranking. `rowid` must address an indexed chunk.
///
/// # Errors
///
/// Fails when scoring fails, or when `rowid` addresses no indexed chunk.
pub fn explain(
    store: &Store,
    embedder: &dyn Embedder,
    query: &str,
    rowid: i64,
    plan: &RetrievalPlan,
    top_k: u32,
) -> crate::Result<ScoreExplanation> {
    let candidates = score_candidates(store, embedder, query, plan, top_k)?;
    let Some(chunk) = store.get_chunks(&[rowid])?.remove(&rowid) else {
        return Err(crate::Error::Store(format!("no chunk with rowid {rowid}")));
    };

    let mut sources = BTreeMap::new();
    sources.insert(
        Source::Semantic,
        source_explanation(
            plan,
            Source::Semantic,
            &candidates.semantic_raw,
            &candidates.prepared,
            rowid,
        ),
    );
    sources.insert(
        Source::Lexical,
        source_explanation(
            plan,
            Source::Lexical,
            &candidates.lexical_raw,
            &candidates.prepared,
            rowid,
        ),
    );
    let combined = candidates.combined.get(&rowid).copied().unwrap_or(0.0);

    let mut all_ranked: Vec<(i64, f64)> = candidates
        .combined
        .iter()
        .filter(|(_, score)| **score > 0.0)
        .map(|(&id, &score)| (id, score))
        .collect();
    all_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let rank = all_ranked
        .iter()
        .position(|&(id, _)| id == rowid)
        .map(|pos| pos + 1);
    let in_results = rank.is_some_and(|rank| rank <= top_k as usize);

    Ok(ScoreExplanation {
        chunk,
        sources,
        combined,
        rank,
        in_results,
        plan: plan.clone(),
        top_k,
    })
}

/// Assemble one source's explanation, deriving its retrieval status.
fn source_explanation(
    plan: &RetrievalPlan,
    source: Source,
    raw: &BTreeMap<i64, f64>,
    prepared: &PreparedScores,
    rowid: i64,
) -> SourceExplanation {
    if !source_enabled(plan, source) {
        return SourceExplanation::new(RetrievalStatus::Disabled, 0);
    }
    if raw.is_empty() {
        return SourceExplanation::new(RetrievalStatus::Empty, 0);
    }
    let Some(&raw_score) = raw.get(&rowid) else {
        return SourceExplanation::new(RetrievalStatus::NotRetrieved, raw.len());
    };
    let prepared = prepared
        .get(&source)
        .and_then(|scores| scores.get(&rowid))
        .copied();
    SourceExplanation {
        status: RetrievalStatus::Matched,
        pool_size: raw.len(),
        raw_score: Some(raw_score),
        prepared_score: prepared,
        pool_rank: pool_rank(raw, rowid),
    }
}

/// Rank of `rowid` within a source pool by raw score, descending.
fn pool_rank(scores: &BTreeMap<i64, f64>, rowid: i64) -> Option<usize> {
    let &own = scores.get(&rowid)?;
    let above = scores.values().filter(|&&v| v > own).count();
    Some(above + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::testing::{FakeEmbedder, file_record, make_chunks};
    use crate::index::ranking::{Fusion, Norm};
    use crate::relpath::WorkspacePath;
    use std::collections::BTreeMap as Weights;

    const DIM: usize = 4;

    /// A min-max + linear plan with the given factors (mirrors `linear_plan`).
    fn linear_plan(semantic: f64, lexical: f64) -> RetrievalPlan {
        let mut weights = Weights::new();
        weights.insert(Source::Semantic, semantic);
        weights.insert(Source::Lexical, lexical);
        RetrievalPlan {
            norm: Some(Norm::MinMax),
            fusion: Fusion::Linear { weights },
        }
    }

    fn rrf_plan() -> RetrievalPlan {
        RetrievalPlan {
            norm: None,
            fusion: Fusion::Rrf { k: 60 },
        }
    }

    /// Two files, three chunks, matching Python's `seeded_store` fixture.
    fn seeded_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store =
            Store::open(&db, "test-model", u32::try_from(DIM).unwrap(), "unicode61").unwrap();
        let (chunks, vectors) =
            make_chunks(&["the quick brown fox jumps", "lazy dog sleeps"], None);
        store
            .replace_file(
                &file_record("a.txt", "blake3:aa", 1),
                &chunks,
                Some(&vectors),
            )
            .unwrap();
        let (chunks, vectors) = make_chunks(&["quantum entanglement in labs"], None);
        store
            .replace_file(
                &file_record("b.md", "blake3:bb", 2),
                &chunks,
                Some(&vectors),
            )
            .unwrap();
        (dir, store)
    }

    fn fake() -> FakeEmbedder {
        FakeEmbedder::default()
    }

    fn run(store: &Store, query: &str, plan: &RetrievalPlan, top_k: u32) -> Vec<SearchHit> {
        search(store, &fake(), query, plan, top_k).unwrap()
    }

    #[test]
    fn search_returns_hits_with_provenance() {
        let (_dir, store) = seeded_store();
        let hits = run(&store, "fox", &linear_plan(0.6, 0.4), 5);
        assert!(!hits.is_empty());
        let hit = &hits[0];
        assert_eq!(hit.primary().path.as_str(), "a.txt");
        assert!(hit.text.contains("fox"));
        assert!(hit.score > 0.0);
    }

    #[test]
    fn search_exposes_page_metadata() {
        let (_dir, mut store) = seeded_store();
        let (chunks, vectors) =
            make_chunks(&["page one content", "page two content"], Some(&[1, 2]));
        store
            .replace_file(
                &file_record("report.pdf", "blake3:cc", 3),
                &chunks,
                Some(&vectors),
            )
            .unwrap();

        let hits = run(&store, "page", &linear_plan(0.6, 0.4), 5);
        let pages: std::collections::BTreeSet<u32> = hits
            .iter()
            .filter(|hit| hit.primary().path.as_str() == "report.pdf")
            .map(|hit| hit.page.expect("pdf chunks carry pages"))
            .collect();
        assert_eq!(pages, [1, 2].into_iter().collect());
    }

    #[test]
    fn sparse_only_weight_uses_bm25() {
        let (_dir, store) = seeded_store();
        let hits = run(&store, "fox", &linear_plan(0.0, 1.0), 5);
        assert!(!hits.is_empty());
        assert!(
            hits.iter()
                .all(|hit| hit.primary().path.as_str() == "a.txt")
        );
    }

    #[test]
    fn dense_only_weight_uses_knn() {
        let (_dir, store) = seeded_store();
        let hits = run(&store, "quantum", &linear_plan(1.0, 0.0), 5);
        assert!(!hits.is_empty());
        // The fake query vector is closest to seq=0 chunks; at least one hit
        // must come from the seeded data regardless of query text.
        assert!(hits.iter().all(|hit| {
            let primary = hit.primary();
            let path = primary.path.as_str();
            path == "a.txt" || path == "b.md"
        }));
    }

    #[test]
    fn search_respects_top_k() {
        let (_dir, store) = seeded_store();
        assert_eq!(run(&store, "fox", &linear_plan(0.6, 0.4), 1).len(), 1);
    }

    #[test]
    fn search_flags_stale_files() {
        let (_dir, mut store) = seeded_store();
        store
            .mark_stale(&WorkspacePath::from_posix("a.txt"))
            .unwrap();

        let hits = run(&store, "fox", &linear_plan(0.6, 0.4), 5);
        assert!(hits.iter().any(SearchHit::stale));
    }

    #[test]
    fn search_empty_query_returns_nothing() {
        let (_dir, store) = seeded_store();
        assert!(run(&store, "   ", &linear_plan(0.6, 0.4), 5).is_empty());
    }

    #[test]
    fn unmatched_query_filters_zero_scores() {
        let (_dir, store) = seeded_store();
        // "zzzzznope" matches nothing lexically: lexical contributes zero, so
        // with lexical-only weights no hit survives the zero-score filter.
        assert!(run(&store, "zzzzznope", &linear_plan(0.0, 1.0), 5).is_empty());
    }

    #[test]
    fn both_factors_zero_raise() {
        let (_dir, store) = seeded_store();
        let err = super::search(&store, &fake(), "fox", &linear_plan(0.0, 0.0), 5).unwrap_err();
        assert!(err.to_string().contains("prospecting tool"));
    }

    #[test]
    fn sum_zero_but_not_both_zero_is_allowed() {
        let (_dir, store) = seeded_store();
        // 0.5 + (-0.5) == 0 is a valid (if unusual) scoring config; only
        // both-zero is rejected.
        let hits = run(&store, "fox", &linear_plan(0.5, -0.5), 5);
        assert!(!hits.is_empty());
    }

    #[test]
    fn rrf_search_ranks_by_position() {
        let (_dir, store) = seeded_store();
        // RRF skips normalization and ranks by reciprocal rank. "fox" matches
        // a.txt lexically and densely, so it should rank first.
        let hits = run(&store, "fox", &rrf_plan(), 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].primary().path.as_str(), "a.txt");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn search_skips_embedder_when_semantic_disabled() {
        let (_dir, store) = seeded_store();
        let embedder = fake();
        let hits = super::search(&store, &embedder, "fox", &linear_plan(0.0, 1.0), 5).unwrap();
        assert!(!hits.is_empty());
        // Semantic is disabled by the zero factor: the query is never embedded.
        assert_eq!(embedder.query_calls.get(), 0);
    }

    #[test]
    fn explain_hit_matches_search_score() {
        let (_dir, store) = seeded_store();
        let plan = linear_plan(0.6, 0.4);
        let hits = run(&store, "fox", &plan, 5);
        let expected = &hits[0];

        let rowid = 1; // first chunk of a.txt
        let explanation = super::explain(&store, &fake(), "fox", rowid, &plan, 5).unwrap();

        assert!((explanation.combined - expected.score).abs() < 1e-12);
        assert_eq!(explanation.chunk.digest, expected.digest);
        assert!(explanation.in_results);
        assert_eq!(explanation.rank, Some(1));
    }

    #[test]
    fn explain_reports_rank_outside_top_k() {
        let (_dir, store) = seeded_store();
        // Rowid 3 (b.md, seq 0) is in the pool but below the top-1 cut.
        let explanation =
            super::explain(&store, &fake(), "fox", 3, &linear_plan(0.6, 0.4), 1).unwrap();
        assert!(explanation.rank.is_some());
        assert!(explanation.rank.unwrap() > 1);
        assert!(!explanation.in_results);
    }

    #[test]
    fn explain_lexical_miss_is_none() {
        let (_dir, store) = seeded_store();
        let explanation =
            super::explain(&store, &fake(), "zzzzznope", 1, &linear_plan(0.6, 0.4), 5).unwrap();
        let lexical = &explanation.sources[&Source::Lexical];
        // The pool is empty, so the source never retrieved anything.
        assert_eq!(lexical.status, RetrievalStatus::Empty);
        assert_eq!(lexical.raw_score, None);
    }

    #[test]
    fn explain_disabled_source_is_none() {
        let (_dir, store) = seeded_store();
        let explanation =
            super::explain(&store, &fake(), "fox", 1, &linear_plan(1.0, 0.0), 5).unwrap();
        let lexical = &explanation.sources[&Source::Lexical];
        assert_eq!(lexical.status, RetrievalStatus::Disabled);
        assert_eq!(lexical.raw_score, None);
    }

    #[test]
    fn explain_zero_combined_not_in_results() {
        let (_dir, store) = seeded_store();
        // "lazy dog sleeps" is dense-ranked last (normalized to 0) and has no
        // lexical match, so its combined score is 0 and it is filtered out.
        let explanation =
            super::explain(&store, &fake(), "fox", 2, &linear_plan(0.6, 0.4), 5).unwrap();
        assert!(explanation.combined.abs() < 1e-12);
        assert_eq!(explanation.rank, None);
        assert!(!explanation.in_results);
    }

    #[test]
    fn explain_unknown_rowid_raises() {
        let (_dir, store) = seeded_store();
        let err =
            super::explain(&store, &fake(), "fox", 999, &linear_plan(0.6, 0.4), 5).unwrap_err();
        assert!(err.to_string().contains("no chunk with rowid 999"));
    }

    #[test]
    fn rrf_explain_skips_norm() {
        let (_dir, store) = seeded_store();
        let plan = rrf_plan();
        let hits = run(&store, "fox", &plan, 5);
        assert!(!hits.is_empty());
        let explanation = super::explain(&store, &fake(), "fox", 1, &plan, 5).unwrap();

        // RRF leaves prepared == raw: no norm ran.
        let semantic = &explanation.sources[&Source::Semantic];
        assert_eq!(semantic.raw_score, semantic.prepared_score);
        assert_eq!(explanation.plan.norm, None);
    }
}
