"""Unit tests for the pluggable ranking operators (norm + fusion).

Hermetic: pure functions over score dicts, no store or network.
"""

from __future__ import annotations

import pytest

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
