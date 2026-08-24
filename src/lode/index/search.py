"""Hybrid retrieval: semantic (dense vec0) + lexical (sparse FTS5) fusion.

Per PLAN D5: both sources are scored, min-max normalized, and combined with
configurable weights. Semantic scores are cosine similarities (vectors are
L2-normalized, so cosine == dot); lexical scores are BM25, normalized
per-query to [0, 1] so the two are comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lode.embeddings.base import Embedder
from lode.index.store import FileStatus, Store

# Retrieve a larger candidate pool than top_k from each source so the fused
# ranking can still reach the best combined result.
CANDIDATE_MULTIPLIER = 4

# Word-ish tokens for the FTS5 query. Each token is quoted so punctuation in
# user queries cannot break the MATCH syntax. CJK runs match as whole tokens
# with the default unicode61 tokenizer; dense retrieval carries that slack.
_FTS_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One fused retrieval result with its provenance."""

    digest: str
    text: str
    path: str
    heading: str
    score: float
    stale: bool
    page: int | None = None


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
    if top_k <= 0 or not query.strip():
        return []
    if semantic_weight == 0 and lexical_weight == 0:
        raise ValueError(
            "You can't discover an ore without a prospecting tool.\n"
            "Hint: at least one of `semantic_factor` and `lexical_factor` must be non-zero."
        )
    pool = top_k * CANDIDATE_MULTIPLIER

    semantic_norm = _minmax(_semantic_scores(store, embedder, query, pool)) if semantic_weight != 0 else {}
    lexical_norm = _minmax(_lexical_scores(store, query, pool)) if lexical_weight != 0 else {}

    combined: dict[int, float] = {}
    for rowid in semantic_norm.keys() | lexical_norm.keys():
        combined[rowid] = semantic_weight * semantic_norm.get(rowid, 0.0) + lexical_weight * lexical_norm.get(
            rowid, 0.0
        )

    # Zero-score rows (no match in either source) carry no signal; dropping
    # them keeps e.g. sparse-only queries from returning unrelated chunks.
    ranked = sorted(
        ((rowid, score) for rowid, score in combined.items() if score > 0.0),
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
                path=chunk.path,
                heading=chunk.heading,
                score=score,
                stale=chunk.file_status is FileStatus.STALE,
                page=chunk.page,
            )
        )
    return hits


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
