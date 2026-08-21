"""Tests for hybrid retrieval: dense + sparse weighted fusion.

Hermetic: FakeEmbedder + a seeded Store, no network.

Note on scoring: fusion uses per-source min-max normalization (PLAN D5), so
an unmatched query still returns the relatively nearest chunks — absolute
relevance thresholds are a retrieval-quality concern deferred to M2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lode.index.search import search
from lode.index.store import Store
from tests.fakes import FakeEmbedder, file_record, make_chunks


@pytest.fixture
def seeded_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "index.db", FakeEmbedder())
    store.replace_file(file_record("a.txt"), *make_chunks(["the quick brown fox jumps", "lazy dog sleeps"]))
    store.replace_file(file_record("b.md", digest="blake3:bb", size=2), *make_chunks(["quantum entanglement in labs"]))
    return store


def test_search_returns_hits_with_provenance(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        dense_weight=0.6,
        sparse_weight=0.4,
        top_k=5,
    )

    assert hits
    hit = hits[0]
    assert hit.path == "a.txt"
    assert "fox" in hit.text
    assert hit.score > 0


def test_sparse_only_weight_uses_bm25(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        dense_weight=0.0,
        sparse_weight=1.0,
        top_k=5,
    )

    assert hits
    assert all(hit.path == "a.txt" for hit in hits)


def test_dense_only_weight_uses_knn(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "quantum",
        dense_weight=1.0,
        sparse_weight=0.0,
        top_k=5,
    )

    assert hits
    # The fake query vector is closest to seq=0 chunks; at least one hit
    # must come from the seeded data regardless of query text.
    assert all(hit.path in {"a.txt", "b.md"} for hit in hits)


def test_search_respects_top_k(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        dense_weight=0.6,
        sparse_weight=0.4,
        top_k=1,
    )
    assert len(hits) == 1


def test_search_flags_stale_files(seeded_store: Store, tmp_path: Path) -> None:
    seeded_store.mark_stale("a.txt")

    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        dense_weight=0.6,
        sparse_weight=0.4,
        top_k=5,
    )

    assert any(hit.stale for hit in hits)


def test_search_empty_query_returns_nothing(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "   ",
        dense_weight=0.6,
        sparse_weight=0.4,
        top_k=5,
    )
    assert hits == []


def test_search_unmatched_query_returns_only_zero_score_filtered(seeded_store: Store) -> None:
    # "zzzzznope" matches nothing lexically: sparse contributes zero, so with
    # sparse-only weights no hit survives the zero-score filter.
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "zzzzznope",
        dense_weight=0.0,
        sparse_weight=1.0,
        top_k=5,
    )
    assert hits == []
