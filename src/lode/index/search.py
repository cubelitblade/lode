"""Hybrid retrieval: semantic (dense vec0) + lexical (sparse FTS5) fusion.

Both sources are scored, then combined by a pluggable ``RetrievalPlan``: a
per-source ``Norm`` (min-max, softmax) followed by a cross-source ``Fusion``
(weighted linear sum, reciprocal rank fusion). Semantic scores are cosine
similarities (L2-normalized, so cosine == dot); lexical scores are BM25.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lode.embeddings.base import Embedder
from lode.index.ranking import LinearFusion, RetrievalPlan
from lode.index.store import ChunkWithPath, FileStatus, PathRef, Store

# Retrieve a larger candidate pool than top_k from each source so the fused
# ranking can still reach the best combined result.
CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One fused retrieval result with its provenance.

    Content is shared by identical files, so a hit carries every path referencing
    it; ``primary`` picks the representative one and ``stale`` reports whether any
    referencing path is outdated.
    """

    digest: str
    text: str
    heading: str
    score: float
    refs: tuple[PathRef, ...]
    page: int | None = None

    @property
    def primary(self) -> PathRef:
        """Representative reference: smallest fresh path, else smallest overall."""
        fresh = [ref for ref in self.refs if ref.status is FileStatus.FRESH]
        return min(fresh or self.refs, key=lambda ref: ref.path)

    @property
    def stale(self) -> bool:
        """Whether any referencing path is stale."""
        return any(ref.status is FileStatus.STALE for ref in self.refs)


@dataclass(frozen=True, slots=True)
class ProspectResult:
    """Aggregated output of a prospect command.

    Carries the query context, the hits, and the library-wide dirty signal
    (``has_stale``, derived from detection: changed or missing files exist),
    distinct from per-hit ``SearchHit.stale``.
    """

    workspace: Path
    query: str
    top_k: int
    hits: list[SearchHit]
    has_stale: bool


@dataclass(frozen=True, slots=True)
class _Candidates:
    """Intermediate per-source scores for one query, before ranking.

    ``*_raw`` are the un-normalized per-source scores (cosine for semantic,
    BM25 for lexical); ``*_prepared`` are their values after the plan's norm
    (or the raw values when the plan skips normalization, e.g. RRF); and
    ``combined`` is the fusion output. ``search`` ranks from ``combined``;
    ``explain`` reads the per-source values for a single chunk.
    """

    semantic_raw: dict[int, float]
    lexical_raw: dict[int, float]
    semantic_prepared: dict[int, float]
    lexical_prepared: dict[int, float]
    combined: dict[int, float]


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    """Why one chunk scored as it did for a query.

    ``*_raw``/``*_prepared`` are ``None`` when that source did not return the
    chunk in its candidate pool (e.g. no lexical match, or the source is
    disabled). ``*_pool_rank`` is the chunk's rank within that source's
    candidate pool (by raw score, descending), or ``None`` when it is not in
    the pool; ``*_pool_size`` is the pool's actual size. ``rank`` is the
    chunk's true rank across all scored rows (not truncated to ``top_k``);
    ``in_results`` reports whether it actually made the returned top-``top_k``.
    ``plan`` is the retrieval path that produced the scores (the flow context
    ``assay`` reads to explain them).
    """

    chunk: ChunkWithPath
    semantic_raw: float | None
    lexical_raw: float | None
    semantic_prepared: float | None
    lexical_prepared: float | None
    semantic_pool_rank: int | None
    lexical_pool_rank: int | None
    semantic_pool_size: int
    lexical_pool_size: int
    combined: float
    rank: int | None
    in_results: bool
    plan: RetrievalPlan
    top_k: int


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    plan: RetrievalPlan,
    top_k: int,
) -> list[SearchHit]:
    """Hybrid search: fusion of semantic and lexical results per ``plan``."""
    candidates = _score_candidates(
        store,
        embedder,
        query,
        plan=plan,
        top_k=top_k,
    )
    if top_k <= 0 or not query.strip():
        return []

    # Zero-score rows (no match in either source) carry no signal; dropping
    # them keeps e.g. sparse-only queries from returning unrelated chunks.
    ranked = sorted(
        ((rowid, score) for rowid, score in candidates.combined.items() if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    if not ranked:
        return []

    chunks = store.get_chunks([rowid for rowid, _ in ranked])
    hits: list[SearchHit] = []
    for rowid, score in ranked:
        chunk = chunks.get(rowid)
        if chunk is None:
            continue
        hits.append(
            SearchHit(
                digest=chunk.digest,
                text=chunk.text,
                heading=chunk.heading,
                score=score,
                refs=chunk.refs,
                page=chunk.page,
            )
        )
    return hits


def _score_candidates(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    plan: RetrievalPlan,
    top_k: int,
) -> _Candidates:
    """Score every candidate chunk for a query, before ranking.

    Shared by ``search`` (which ranks from ``combined``) and ``explain``
    (which reads the per-source values for one chunk), so the two never
    drift apart. An empty query or non-positive ``top_k`` yields empty
    candidates; a linear plan with both weights zero raises.
    """
    if top_k <= 0 or not query.strip():
        return _Candidates({}, {}, {}, {}, {})
    if isinstance(plan.fusion, LinearFusion):
        weights = plan.fusion.weights
        if weights.get("semantic", 0.0) == 0 and weights.get("lexical", 0.0) == 0:
            raise ValueError(
                "You can't discover an ore without a prospecting tool.\n"
                "Hint: at least one of `semantic_factor` and `lexical_factor` must be non-zero."
            )
    pool = top_k * CANDIDATE_MULTIPLIER

    semantic_raw = _semantic_scores(store, embedder, query, pool) if _source_enabled(plan, "semantic") else {}
    lexical_raw = _lexical_scores(store, query, pool) if _source_enabled(plan, "lexical") else {}
    prepared = _prepare(plan, {"semantic": semantic_raw, "lexical": lexical_raw})
    semantic_prepared = prepared["semantic"]
    lexical_prepared = prepared["lexical"]

    combined = plan.fusion.fuse(prepared)
    return _Candidates(semantic_raw, lexical_raw, semantic_prepared, lexical_prepared, combined)


def _source_enabled(plan: RetrievalPlan, source: str) -> bool:
    """Whether a source is queried at all.

    For linear fusion a zero weight disables the source (short-circuit, kept
    from the pre-plan behaviour); RRF always queries every source.
    """
    if isinstance(plan.fusion, LinearFusion):
        return plan.fusion.weights.get(source, 0.0) != 0
    return True


def _prepare(plan: RetrievalPlan, raw: Mapping[str, dict[int, float]]) -> dict[str, dict[int, float]]:
    """Apply the plan's norm per source, or pass raw through when skipped."""
    if plan.norm is None:
        return dict(raw)
    return {source: plan.norm.normalize(scores) for source, scores in raw.items()}


def explain(
    store: Store,
    embedder: Embedder,
    query: str,
    rowid: int,
    *,
    plan: RetrievalPlan,
    top_k: int,
) -> ScoreExplanation:
    """Explain why one chunk scored as it did for a query.

    Reuses the same candidate scoring as ``search`` so the explanation always
    matches the real ranking. ``rowid`` must address an indexed chunk.
    """
    candidates = _score_candidates(
        store,
        embedder,
        query,
        plan=plan,
        top_k=top_k,
    )
    chunk = store.get_chunks([rowid]).get(rowid)
    if chunk is None:
        raise ValueError(f"no chunk with rowid {rowid}")

    semantic_raw = candidates.semantic_raw.get(rowid)
    lexical_raw = candidates.lexical_raw.get(rowid)
    semantic_prepared = candidates.semantic_prepared.get(rowid)
    lexical_prepared = candidates.lexical_prepared.get(rowid)
    combined = candidates.combined.get(rowid, 0.0)

    semantic_pool_rank = _pool_rank(candidates.semantic_raw, rowid)
    lexical_pool_rank = _pool_rank(candidates.lexical_raw, rowid)
    semantic_pool_size = len(candidates.semantic_raw)
    lexical_pool_size = len(candidates.lexical_raw)

    all_ranked = sorted(
        ((rid, score) for rid, score in candidates.combined.items() if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )
    rank = next((i + 1 for i, (rid, _) in enumerate(all_ranked) if rid == rowid), None)
    in_results = rank is not None and rank <= top_k

    return ScoreExplanation(
        chunk=chunk,
        semantic_raw=semantic_raw,
        lexical_raw=lexical_raw,
        semantic_prepared=semantic_prepared,
        lexical_prepared=lexical_prepared,
        semantic_pool_rank=semantic_pool_rank,
        lexical_pool_rank=lexical_pool_rank,
        semantic_pool_size=semantic_pool_size,
        lexical_pool_size=lexical_pool_size,
        combined=combined,
        rank=rank,
        in_results=in_results,
        plan=plan,
        top_k=top_k,
    )


def _pool_rank(scores: dict[int, float], rowid: int) -> int | None:
    """Rank of ``rowid`` within a source pool by raw score, descending."""
    if rowid not in scores:
        return None
    ordered = sorted(scores.values(), reverse=True)
    return ordered.index(scores[rowid]) + 1


def _semantic_scores(store: Store, embedder: Embedder, query: str, k: int) -> dict[int, float]:
    vector = embedder.embed_query(query)
    return {match.rowid: _cosine(match.distance) for match in store.dense_search(vector, k)}


def _lexical_scores(store: Store, query: str, k: int) -> dict[int, float]:
    return {match.rowid: match.score for match in store.sparse_search(query, k)}


def _cosine(distance: float) -> float:
    """Cosine similarity from L2 distance over L2-normalized vectors.

    For normalized vectors, d^2 = 2 - 2*cos, so cos = 1 - d^2/2.
    """
    return 1.0 - (distance * distance) / 2.0
