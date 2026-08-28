"""Tests for retrieval explanations (why a chunk scored the way it did).

Hermetic: FakeEmbedder + a seeded Store, no network.
"""

from __future__ import annotations

import pytest

from lode.index import Store
from lode.index.explanation import RetrievalStatus
from lode.index.ranking import RetrievalPlan, RrfFusion
from lode.index.search import explain, search
from lode.ingestion import chunk_digest
from tests.conftest import linear_plan
from tests.fakes import FakeEmbedder


def _rowid(store: Store, text: str) -> int:
    """Rowid of the single chunk whose digest addresses ``text``."""
    prefix = chunk_digest(text).removeprefix("blake3:")
    return store.find_chunk_rowids(prefix)[0]


def test_explain_hit_matches_search_score(seeded_store: Store) -> None:
    text = "the quick brown fox jumps"
    rowid = _rowid(seeded_store, text)
    explanation = explain(
        seeded_store,
        FakeEmbedder(),
        "fox",
        rowid,
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )

    assert explanation.in_results
    assert explanation.rank is not None
    assert explanation.combined > 0
    assert explanation.sources["semantic"].raw_score is not None
    assert explanation.sources["lexical"].raw_score is not None

    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )
    hit = next(h for h in hits if h.digest == chunk_digest(text))
    assert explanation.combined == pytest.approx(hit.score)


def test_explain_reports_rank_outside_top_k(seeded_store: Store) -> None:
    # b.md is dense-tied with a.txt seq0 but has no lexical match, so it ranks
    # second; with top_k=1 it is not returned.
    text = "quantum entanglement in labs"
    rowid = _rowid(seeded_store, text)
    explanation = explain(
        seeded_store,
        FakeEmbedder(),
        "fox",
        rowid,
        plan=linear_plan(0.6, 0.4),
        top_k=1,
    )

    assert explanation.rank == 2
    assert not explanation.in_results
    assert explanation.combined > 0


def test_explain_lexical_miss_is_none(seeded_store: Store) -> None:
    # "lazy dog sleeps" has no "fox" token, so its lexical score is absent
    # even though it is in the dense candidate pool.
    text = "lazy dog sleeps"
    rowid = _rowid(seeded_store, text)
    explanation = explain(
        seeded_store,
        FakeEmbedder(),
        "fox",
        rowid,
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )

    assert explanation.sources["lexical"].status is RetrievalStatus.NOT_RETRIEVED
    assert explanation.sources["lexical"].raw_score is None
    assert explanation.sources["semantic"].raw_score is not None


def test_explain_disabled_source_is_none(seeded_store: Store) -> None:
    text = "the quick brown fox jumps"
    rowid = _rowid(seeded_store, text)
    explanation = explain(
        seeded_store,
        FakeEmbedder(),
        "fox",
        rowid,
        plan=linear_plan(0.0, 1.0),
        top_k=5,
    )

    assert explanation.sources["semantic"].status is RetrievalStatus.DISABLED
    assert explanation.sources["semantic"].raw_score is None
    assert explanation.sources["lexical"].raw_score is not None


def test_explain_zero_combined_not_in_results(seeded_store: Store) -> None:
    # "lazy dog sleeps" is dense-ranked last (normalized to 0) and has no
    # lexical match, so its combined score is 0 and it is filtered out.
    text = "lazy dog sleeps"
    rowid = _rowid(seeded_store, text)
    explanation = explain(
        seeded_store,
        FakeEmbedder(),
        "fox",
        rowid,
        plan=linear_plan(0.6, 0.4),
        top_k=5,
    )

    assert explanation.combined == 0.0
    assert explanation.rank is None
    assert not explanation.in_results


def test_explain_unknown_rowid_raises(seeded_store: Store) -> None:
    with pytest.raises(ValueError, match="no chunk with rowid"):
        explain(
            seeded_store,
            FakeEmbedder(),
            "fox",
            9999,
            plan=linear_plan(0.6, 0.4),
            top_k=5,
        )


def test_rrf_explain_skips_norm(seeded_store: Store) -> None:
    text = "the quick brown fox jumps"
    rowid = _rowid(seeded_store, text)
    plan = RetrievalPlan(norm=None, fusion=RrfFusion(k=60))
    explanation = explain(
        seeded_store,
        FakeEmbedder(),
        "fox",
        rowid,
        plan=plan,
        top_k=5,
    )

    assert explanation.plan.norm is None
    # RRF skips normalization: prepared scores equal the raw ones.
    assert explanation.sources["semantic"].prepared_score == explanation.sources["semantic"].raw_score
    assert explanation.sources["lexical"].prepared_score == explanation.sources["lexical"].raw_score
    assert explanation.in_results
