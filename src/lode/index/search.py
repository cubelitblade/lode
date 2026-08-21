"""Hybrid retrieval: dense (vec0) + sparse (FTS5) with weighted fusion.

Per PLAN D5: both sources are scored, min-max normalized, and combined with
configurable weights. Dense scores are cosine similarities (vectors are
L2-normalized, so cosine == dot); sparse scores are BM25, normalized
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

    chunk_id: str
    text: str
    path: str
    heading: str
    score: float
    stale: bool


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    dense_weight: float,
    sparse_weight: float,
    top_k: int,
) -> list[SearchHit]:
    """Hybrid search: weighted fusion of dense and sparse results."""
    if top_k <= 0 or not query.strip():
        return []
    pool = top_k * CANDIDATE_MULTIPLIER

    dense_scores = _dense_scores(store, embedder, query, pool)
    sparse_scores = _sparse_scores(store, query, pool)

    dense_norm = _minmax(dense_scores)
    sparse_norm = _minmax(sparse_scores)

    combined: dict[int, float] = {}
    for rowid in dense_norm.keys() | sparse_norm.keys():
        combined[rowid] = dense_weight * dense_norm.get(rowid, 0.0) + sparse_weight * sparse_norm.get(rowid, 0.0)

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
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                path=chunk.path,
                heading=chunk.heading,
                score=score,
                stale=chunk.file_status is FileStatus.STALE,
            )
        )
    return hits


def _dense_scores(store: Store, embedder: Embedder, query: str, k: int) -> dict[int, float]:
    vector = embedder.embed_query(query)
    return {rowid: _cosine(distance) for rowid, distance in store.dense_search(vector, k)}


def _sparse_scores(store: Store, query: str, k: int) -> dict[int, float]:
    fts_query = _fts_query(query)
    if not fts_query:
        return {}
    return dict(store.sparse_search(fts_query, k))


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
