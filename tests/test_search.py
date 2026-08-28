"""Tests for hybrid retrieval: dense + sparse weighted fusion.

Hermetic: FakeEmbedder + a seeded Store, no network.

Note on scoring: fusion uses per-source min-max normalization (PLAN D5), so
an unmatched query still returns the relatively nearest chunks — absolute
relevance thresholds are a retrieval-quality concern deferred to M2.
"""

from __future__ import annotations

import pytest

from lode.index import Store
from lode.index.ranking import RetrievalPlan, RrfFusion
from lode.index.search import search
from tests.conftest import linear_plan
from tests.fakes import FakeEmbedder, file_record, make_chunks


def test_search_returns_hits_with_provenance(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )

    assert hits
    hit = hits[0]
    assert hit.primary.path == "a.txt"
    assert "fox" in hit.text
    assert hit.score > 0


def test_search_exposes_page_metadata(seeded_store: Store) -> None:
    seeded_store.replace_file(
        file_record("report.pdf", digest="blake3:cc", size=3),
        *make_chunks(["page one content", "page two content"], pages=[1, 2]),
    )

    hits = search(
        seeded_store,
        FakeEmbedder(),
        "page",
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )

    pdf_hits = [hit for hit in hits if hit.primary.path == "report.pdf"]
    assert pdf_hits
    assert {hit.page for hit in pdf_hits} == {1, 2}


def test_sparse_only_weight_uses_bm25(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=linear_plan(0.0, 1.0),
        top_k=5,
    )

    assert hits
    assert all(hit.primary.path == "a.txt" for hit in hits)


def test_dense_only_weight_uses_knn(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "quantum",
        plan=linear_plan(1.0, 0.0),
        top_k=5,
    )

    assert hits
    # The fake query vector is closest to seq=0 chunks; at least one hit
    # must come from the seeded data regardless of query text.
    assert all(hit.primary.path in {"a.txt", "b.md"} for hit in hits)


def test_search_respects_top_k(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=linear_plan(0.6, 0.4),
        top_k=1,
    )
    assert len(hits) == 1


def test_search_flags_stale_files(seeded_store: Store) -> None:
    seeded_store.mark_stale("a.txt")

    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )

    assert any(hit.stale for hit in hits)


def test_search_empty_query_returns_nothing(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "   ",
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )
    assert hits == []


def test_search_unmatched_query_returns_only_zero_score_filtered(seeded_store: Store) -> None:
    # "zzzzznope" matches nothing lexically: lexical contributes zero, so with
    # lexical-only weights no hit survives the zero-score filter.
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "zzzzznope",
        plan=linear_plan(0.0, 1.0),
        top_k=5,
    )
    assert hits == []


def test_both_factors_zero_raise(seeded_store: Store) -> None:
    with pytest.raises(ValueError, match="prospecting tool"):
        search(
            seeded_store,
            FakeEmbedder(),
            "fox",
            plan=linear_plan(0.0, 0.0),
            top_k=5,
        )


def test_sum_zero_but_not_both_zero_is_allowed(seeded_store: Store) -> None:
    # e.g. 0.5 + (-0.5) == 0 is a valid (if unusual) scoring config; only
    # both-zero is rejected.
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=linear_plan(0.5, -0.5),
        top_k=5,
    )
    assert isinstance(hits, list)


def test_rrf_search_ranks_by_position(seeded_store: Store) -> None:
    # RRF skips normalization and ranks by reciprocal rank. "fox" matches
    # a.txt lexically and densely, so it should rank first.
    plan = RetrievalPlan(norm=None, fusion=RrfFusion(k=60))
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=plan,
        top_k=5,
    )

    assert hits
    assert hits[0].primary.path == "a.txt"
    assert hits[0].score > 0
