"""Hybrid retrieval: semantic (dense vec0) + lexical (sparse FTS5) fusion.

Both sources are scored, min-max normalized, and combined with configurable
weights. Semantic scores are cosine similarities (L2-normalized, so cosine
== dot); lexical scores are BM25, normalized per-query to [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lode.embeddings.base import Embedder
from lode.index.store import ChunkWithPath, FileStatus, PathRef, Store

# Retrieve a larger candidate pool than top_k from each source so the fused
# ranking can still reach the best combined result.
CANDIDATE_MULTIPLIER = 4

# Word-ish tokens for the FTS5 query. Each token is quoted so punctuation in
# user queries cannot break the MATCH syntax. CJK runs match as whole tokens
# with the default unicode61 tokenizer; dense retrieval carries that slack.
_FTS_TOKEN = re.compile(r"\w+")


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
    BM25 for lexical); ``*_norm`` are their min-max normalized forms; and
    ``combined`` is the weighted fusion. ``search`` ranks from ``combined``;
    ``explain`` reads the per-source values for a single chunk.
    """

    semantic_raw: dict[int, float]
    lexical_raw: dict[int, float]
    semantic_norm: dict[int, float]
    lexical_norm: dict[int, float]
    combined: dict[int, float]


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    """Why one chunk scored as it did for a query.

    ``*_raw``/``*_norm`` are ``None`` when that source did not return the
    chunk in its candidate pool (e.g. no lexical match, or the weight is
    zero so the source was not queried). ``*_pool_rank`` is the chunk's rank
    within that source's candidate pool (by raw score, descending), or
    ``None`` when it is not in the pool; ``*_pool_size`` is the pool's actual
    size. ``rank`` is the chunk's true rank across all scored rows (not
    truncated to ``top_k``); ``in_results`` reports whether it actually made
    the returned top-``top_k``.
    """

    chunk: ChunkWithPath
    semantic_raw: float | None
    lexical_raw: float | None
    semantic_norm: float | None
    lexical_norm: float | None
    semantic_pool_rank: int | None
    lexical_pool_rank: int | None
    semantic_pool_size: int
    lexical_pool_size: int
    combined: float
    rank: int | None
    in_results: bool
    semantic_weight: float
    lexical_weight: float
    top_k: int


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    semantic_weight: float,
    lexical_weight: float,
    top_k: int,
) -> list[SearchHit]:
    """Hybrid search: weighted fusion of semantic and lexical results."""
    candidates = _score_candidates(
        store,
        embedder,
        query,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
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
    semantic_weight: float,
    lexical_weight: float,
    top_k: int,
) -> _Candidates:
    """Score every candidate chunk for a query, before ranking.

    Shared by ``search`` (which ranks from ``combined``) and ``explain``
    (which reads the per-source values for one chunk), so the two never
    drift apart. An empty query or non-positive ``top_k`` yields empty
    candidates; both-zero weights raise.
    """
    if top_k <= 0 or not query.strip():
        return _Candidates({}, {}, {}, {}, {})
    if semantic_weight == 0 and lexical_weight == 0:
        raise ValueError(
            "You can't discover an ore without a prospecting tool.\n"
            "Hint: at least one of `semantic_factor` and `lexical_factor` must be non-zero."
        )
    pool = top_k * CANDIDATE_MULTIPLIER

    semantic_raw = _semantic_scores(store, embedder, query, pool) if semantic_weight != 0 else {}
    lexical_raw = _lexical_scores(store, query, pool) if lexical_weight != 0 else {}
    semantic_norm = _minmax(semantic_raw)
    lexical_norm = _minmax(lexical_raw)

    combined: dict[int, float] = {}
    for rowid in semantic_norm.keys() | lexical_norm.keys():
        combined[rowid] = semantic_weight * semantic_norm.get(rowid, 0.0) + lexical_weight * lexical_norm.get(
            rowid, 0.0
        )
    return _Candidates(semantic_raw, lexical_raw, semantic_norm, lexical_norm, combined)


def explain(
    store: Store,
    embedder: Embedder,
    query: str,
    rowid: int,
    *,
    semantic_weight: float,
    lexical_weight: float,
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
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        top_k=top_k,
    )
    chunk = store.get_chunks([rowid]).get(rowid)
    if chunk is None:
        raise ValueError(f"no chunk with rowid {rowid}")

    semantic_raw = candidates.semantic_raw.get(rowid)
    lexical_raw = candidates.lexical_raw.get(rowid)
    semantic_norm = candidates.semantic_norm.get(rowid)
    lexical_norm = candidates.lexical_norm.get(rowid)
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
        semantic_norm=semantic_norm,
        lexical_norm=lexical_norm,
        semantic_pool_rank=semantic_pool_rank,
        lexical_pool_rank=lexical_pool_rank,
        semantic_pool_size=semantic_pool_size,
        lexical_pool_size=lexical_pool_size,
        combined=combined,
        rank=rank,
        in_results=in_results,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
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
    fts_query = _fts_query(query)
    if not fts_query:
        return {}
    return {match.rowid: match.score for match in store.sparse_search(fts_query, k)}


def _cosine(distance: float) -> float:
    """Cosine similarity from L2 distance over L2-normalized vectors.

    For normalized vectors, d^2 = 2 - 2*cos, so cos = 1 - d^2/2.
    """
    return 1.0 - (distance * distance) / 2.0


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize to [0, 1]; a single value maps to 1.0."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span == 0.0:
        return {rowid: 1.0 for rowid in scores}
    return {rowid: (score - lo) / span for rowid, score in scores.items()}


def _fts_query(text: str) -> str:
    tokens = _FTS_TOKEN.findall(text)
    return " OR ".join(f'"{token}"' for token in tokens)
