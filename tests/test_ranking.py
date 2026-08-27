"""Unit tests for the pluggable ranking operators (norm + fusion).

Hermetic: pure functions over score dicts, no store or network.
"""

from __future__ import annotations

import pytest

from lode.index.explanation import RetrievalStatus, SourceExplanation
from lode.index.ranking import LinearFusion, MinmaxNorm, RrfFusion, SoftmaxNorm


def test_minmax_normalizes_to_unit_interval() -> None:
    norm = MinmaxNorm()
    result = norm.normalize({1: 0.0, 2: 5.0, 3: 10.0})
    assert result == {1: 0.0, 2: 0.5, 3: 1.0}


def test_minmax_single_value_maps_to_one() -> None:
    norm = MinmaxNorm()
    assert norm.normalize({1: 3.0}) == {1: 1.0}


def test_minmax_empty_is_empty() -> None:
    assert MinmaxNorm().normalize({}) == {}


def test_minmax_flat_scores_map_to_one() -> None:
    # All scores equal -> span is zero -> every value maps to 1.0.
    assert MinmaxNorm().normalize({1: 2.0, 2: 2.0}) == {1: 1.0, 2: 1.0}


def test_softmax_sums_to_one() -> None:
    norm = SoftmaxNorm(temperature=1.0)
    result = norm.normalize({1: 1.0, 2: 2.0, 3: 3.0})
    assert sum(result.values()) == pytest.approx(1.0)
    # Higher raw score -> higher normalized score (monotonic).
    assert result[3] > result[2] > result[1]


def test_softmax_temperature_flattens() -> None:
    hot = SoftmaxNorm(temperature=10.0).normalize({1: 1.0, 2: 2.0})
    cold = SoftmaxNorm(temperature=0.1).normalize({1: 1.0, 2: 2.0})
    # Higher temperature spreads scores toward uniform.
    assert (hot[2] - hot[1]) < (cold[2] - cold[1])


def test_softmax_empty_is_empty() -> None:
    assert SoftmaxNorm().normalize({}) == {}


def test_linear_fusion_weighted_sum() -> None:
    fusion = LinearFusion(weights={"semantic": 0.7, "lexical": 0.3})
    prepared = {"semantic": {1: 1.0, 2: 0.5}, "lexical": {2: 1.0, 3: 0.5}}
    result = fusion.fuse(prepared)
    assert result[1] == pytest.approx(0.7)
    assert result[2] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
    assert result[3] == pytest.approx(0.3 * 0.5)


def test_linear_fusion_unknown_source_weight_zero() -> None:
    fusion = LinearFusion(weights={"semantic": 1.0})
    result = fusion.fuse({"lexical": {1: 5.0}})
    assert result[1] == 0.0


def test_rrf_fusion_reciprocal_rank() -> None:
    fusion = RrfFusion(k=60)
    prepared = {"semantic": {1: 10.0, 2: 5.0}, "lexical": {2: 9.0, 3: 1.0}}
    result = fusion.fuse(prepared)
    # semantic ranks: 1 -> pos1, 2 -> pos2; lexical ranks: 2 -> pos1, 3 -> pos2.
    assert result[1] == pytest.approx(1.0 / 61.0)
    assert result[2] == pytest.approx(1.0 / 62.0 + 1.0 / 61.0)
    assert result[3] == pytest.approx(1.0 / 62.0)


def test_rrf_fusion_uses_custom_k() -> None:
    fusion = RrfFusion(k=10)
    result = fusion.fuse({"semantic": {1: 1.0}})
    assert result[1] == pytest.approx(1.0 / 11.0)


def _matched(
    *,
    raw: float,
    prepared: float,
    pool_rank: int,
    pool_size: int = 40,
) -> SourceExplanation:
    return SourceExplanation(
        status=RetrievalStatus.MATCHED,
        pool_size=pool_size,
        raw_score=raw,
        prepared_score=prepared,
        pool_rank=pool_rank,
    )


def _sources() -> dict[str, SourceExplanation]:
    return {
        "semantic": _matched(raw=0.9, prepared=1.0, pool_rank=1),
        "lexical": _matched(raw=-5.0, prepared=0.5, pool_rank=3),
    }


def test_linear_explain_ranking_factors_are_contributions() -> None:
    fusion = LinearFusion(weights={"semantic": 0.7, "lexical": 0.3})
    expl = fusion.explain(_sources(), combined=0.85, norm=MinmaxNorm())
    assert expl.ranking_factors["semantic"].metric == "contribution"
    assert expl.ranking_factors["semantic"].value == pytest.approx(0.7)
    assert expl.ranking_factors["lexical"].value == pytest.approx(0.15)


def test_linear_explain_evidence_carries_norm_and_weight() -> None:
    fusion = LinearFusion(weights={"semantic": 0.7, "lexical": 0.3})
    expl = fusion.explain(_sources(), combined=0.85, norm=MinmaxNorm())
    block = expl.evidence["semantic"]
    assert block.rank == 1
    assert block.raw_score == pytest.approx(0.9)
    assert block.prepared_score == pytest.approx(1.0)
    assert block.normalization == "min-max"
    assert block.weight == pytest.approx(0.7)
    assert block.contribution == pytest.approx(0.7)


def test_linear_explain_formula_substitutes_numbers() -> None:
    fusion = LinearFusion(weights={"semantic": 0.7, "lexical": 0.3})
    expl = fusion.explain(_sources(), combined=0.85, norm=MinmaxNorm())
    assert expl.formula.method_label == "Linear"
    assert expl.formula.symbolic_terms == "semantic \u00d7 0.7 + lexical \u00d7 0.3"
    assert expl.formula.value_terms == "1.0000 \u00d7 0.7 + 0.5000 \u00d7 0.3"
    assert expl.formula.result == pytest.approx(0.85)
    assert expl.formula.missing_note is None


def test_linear_explain_skips_non_matched_sources() -> None:
    sources = _sources()
    sources["lexical"] = SourceExplanation(
        status=RetrievalStatus.DISABLED, pool_size=0, raw_score=None, prepared_score=None, pool_rank=None
    )
    fusion = LinearFusion(weights={"semantic": 0.7, "lexical": 0.3})
    expl = fusion.explain(sources, combined=0.7, norm=MinmaxNorm())
    assert set(expl.ranking_factors) == {"semantic"}
    assert set(expl.evidence) == {"semantic"}
    assert expl.formula.symbolic_terms == "semantic \u00d7 0.7"


def test_rrf_explain_ranking_factors_are_ranks() -> None:
    fusion = RrfFusion(k=60)
    expl = fusion.explain(_sources(), combined=0.05)
    assert expl.ranking_factors["semantic"].metric == "rank"
    assert expl.ranking_factors["semantic"].value == 1
    assert expl.ranking_factors["lexical"].value == 3


def test_rrf_explain_evidence_is_empty() -> None:
    fusion = RrfFusion(k=60)
    expl = fusion.explain(_sources(), combined=0.05)
    assert expl.evidence == {}


def test_rrf_explain_formula_uses_ranks() -> None:
    fusion = RrfFusion(k=60)
    expl = fusion.explain(_sources(), combined=0.05)
    assert expl.formula.method_label == "RRF (k=60)"
    assert expl.formula.symbolic_terms == "1 \u00f7 (60 + semantic_rank) + 1 \u00f7 (60 + lexical_rank)"
    assert expl.formula.value_terms == "1 \u00f7 61 + 1 \u00f7 63"
    assert expl.formula.result == pytest.approx(0.05)
    assert expl.formula.missing_note is None


def test_rrf_explain_missing_note_lists_non_matched() -> None:
    sources = _sources()
    sources["lexical"] = SourceExplanation(
        status=RetrievalStatus.EMPTY, pool_size=0, raw_score=None, prepared_score=None, pool_rank=None
    )
    fusion = RrfFusion(k=60)
    expl = fusion.explain(sources, combined=1.0 / 61.0)
    assert set(expl.ranking_factors) == {"semantic"}
    assert expl.formula.missing_note == "lexical: no results"
