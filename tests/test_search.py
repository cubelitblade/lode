"""Tests for hybrid retrieval: dense + sparse weighted fusion.

Hermetic: FakeEmbedder + a seeded Store, no network.

Note on scoring: fusion uses per-source min-max normalization (PLAN D5), so
an unmatched query still returns the relatively nearest chunks — absolute
relevance thresholds are a retrieval-quality concern deferred to M2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lode.index import Store
from lode.index.explanation import RetrievalStatus
from lode.index.ranking import LinearFusion, MinmaxNorm, RetrievalPlan, RrfFusion
from lode.index.search import explain, search
from lode.ingestion import chunk_digest
from tests.fakes import FakeEmbedder, file_record, make_chunks


def _linear_plan(semantic: float, lexical: float) -> RetrievalPlan:
    """A min-max + linear plan with the given weights (the default shape)."""
    return RetrievalPlan(
        norm=MinmaxNorm(),
        fusion=LinearFusion(weights={"semantic": semantic, "lexical": lexical}),
    )


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
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.0, 1.0),
        top_k=5,
    )

    assert hits
    assert all(hit.primary.path == "a.txt" for hit in hits)


def test_dense_only_weight_uses_knn(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "quantum",
        plan=_linear_plan(1.0, 0.0),
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
        plan=_linear_plan(0.6, 0.4),
        top_k=1,
    )
    assert len(hits) == 1


def test_search_flags_stale_files(seeded_store: Store, tmp_path: Path) -> None:
    seeded_store.mark_stale("a.txt")

    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=_linear_plan(0.6, 0.4),
        top_k=5,
    )

    assert any(hit.stale for hit in hits)


def test_search_empty_query_returns_nothing(seeded_store: Store) -> None:
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "   ",
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.0, 1.0),
        top_k=5,
    )
    assert hits == []


def test_both_factors_zero_raise(seeded_store: Store) -> None:
    with pytest.raises(ValueError, match="prospecting tool"):
        search(
            seeded_store,
            FakeEmbedder(),
            "fox",
            plan=_linear_plan(0.0, 0.0),
            top_k=5,
        )


def test_sum_zero_but_not_both_zero_is_allowed(seeded_store: Store) -> None:
    # e.g. 0.5 + (-0.5) == 0 is a valid (if unusual) scoring config; only
    # both-zero is rejected.
    hits = search(
        seeded_store,
        FakeEmbedder(),
        "fox",
        plan=_linear_plan(0.5, -0.5),
        top_k=5,
    )
    assert isinstance(hits, list)


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
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.6, 0.4),
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
        plan=_linear_plan(0.0, 1.0),
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
        plan=_linear_plan(0.6, 0.4),
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
            plan=_linear_plan(0.6, 0.4),
            top_k=5,
        )


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
