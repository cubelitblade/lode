"""Pluggable retrieval ranking: per-source normalization + cross-source fusion.

The scoring pipeline is a fixed two-stage shape — each source is scored, then
the per-source scores are combined — but the operators are swappable. ``Norm``
transforms one source's scores within its own domain (min-max, softmax);
``Fusion`` merges the prepared per-source scores into one combined score
(weighted linear sum, reciprocal rank fusion).

``RetrievalPlan`` is the assembly-time product that freezes which operators a
query actually runs. It doubles as the "flow context" that ``assay`` reads to
explain a score: it carries the active ``norm`` (``None`` when fusion is RRF,
which ranks by position and so skips normalization) and the active ``fusion``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from lode.index.explanation import (
    EvidenceBlock,
    FormulaComponents,
    FusionExplanation,
    RankingFactor,
    RetrievalStatus,
    SourceExplanation,
)


class Norm(Protocol):
    """Per-source score transform, applied before fusion.

    Implementations map one source's raw scores to a comparable domain. The
    transform must be monotonic so it never changes the source's internal
    ranking (RRF relies on this to skip normalization safely).
    """

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> Mapping[str, object]:
        """Human-readable operator parameters (e.g. ``temperature``).

        Empty for parameterless operators; consumed by explanation output so
        renderers never inspect concrete operator types.
        """
        ...

    def normalize(self, scores: dict[int, float]) -> dict[int, float]: ...


@dataclass(frozen=True, slots=True)
class MinmaxNorm:
    """Min-max normalize to [0, 1]; a single value maps to 1.0."""

    name: str = "min-max"

    @property
    def params(self) -> Mapping[str, object]:
        return {}

    def normalize(self, scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        lo = min(scores.values())
        hi = max(scores.values())
        span = hi - lo
        if span == 0.0:
            return {rowid: 1.0 for rowid in scores}
        return {rowid: (score - lo) / span for rowid, score in scores.items()}


@dataclass(frozen=True, slots=True)
class SoftmaxNorm:
    """Softmax over scores scaled by ``temperature``.

    Higher temperature flattens the distribution toward uniform; lower
    temperature sharpens it toward the top score. The transform is monotonic
    in each score, so it preserves the source's internal ranking.
    """

    temperature: float = 1.0
    name: str = "softmax"

    @property
    def params(self) -> Mapping[str, object]:
        return {"temperature": self.temperature}

    def normalize(self, scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        scaled = [score / self.temperature for score in scores.values()]
        max_scaled = max(scaled)
        # Subtract the max before exp to keep the sum finite for large inputs.
        exps = [math.exp(value - max_scaled) for value in scaled]
        total = sum(exps)
        return {rowid: exp / total for rowid, exp in zip(scores.keys(), exps, strict=True)}


class Fusion(Protocol):
    """Cross-source merge of prepared per-source scores into one combined score."""

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> Mapping[str, object]:
        """Human-readable operator parameters (e.g. ``k``, ``weights``)."""
        ...

    def fuse(self, prepared: Mapping[str, dict[int, float]]) -> dict[int, float]: ...

    def explain(
        self,
        sources: Mapping[str, SourceExplanation],
        combined: float,
        norm: Norm | None = None,
    ) -> FusionExplanation:
        """Render-ready explanation of how this fusion produced ``combined``.

        The render layer consumes this without branching on concrete operator
        types. ``norm`` is provided so Linear can include the normalization
        name in evidence blocks; RRF ignores it.
        """
        ...


@dataclass(frozen=True, slots=True)
class LinearFusion:
    """Weighted sum of per-source scores: ``sum(w_s * prepared[s][rowid])``."""

    weights: Mapping[str, float]
    name: str = "linear"

    @property
    def params(self) -> Mapping[str, object]:
        return {"weights": dict(self.weights)}

    def fuse(self, prepared: Mapping[str, dict[int, float]]) -> dict[int, float]:
        combined: dict[int, float] = {}
        for source, scores in prepared.items():
            weight = self.weights.get(source, 0.0)
            for rowid, score in scores.items():
                combined[rowid] = combined.get(rowid, 0.0) + weight * score
        return combined

    def explain(
        self,
        sources: Mapping[str, SourceExplanation],
        combined: float,
        norm: Norm | None = None,
    ) -> FusionExplanation:
        norm_name = norm.name if norm is not None else None
        ranking_factors: dict[str, RankingFactor] = {}
        evidence: dict[str, EvidenceBlock] = {}
        symbolic_parts: list[str] = []
        value_parts: list[str] = []

        for name, source in sources.items():
            if source.status is not RetrievalStatus.MATCHED:
                continue
            weight = self.weights.get(name, 0.0)
            prepared = source.prepared_score or 0.0
            contribution = prepared * weight
            ranking_factors[name] = RankingFactor("contribution", contribution)
            evidence[name] = EvidenceBlock(
                rank=source.pool_rank,
                raw_score=source.raw_score,
                prepared_score=source.prepared_score,
                normalization=norm_name,
                weight=weight,
                contribution=contribution,
            )
            symbolic_parts.append(f"{name} \u00d7 {weight}")
            value_parts.append(f"{source.prepared_score:.4f} \u00d7 {weight}")

        formula = FormulaComponents(
            method_label="Linear",
            symbolic_terms=" + ".join(symbolic_parts),
            value_terms=" + ".join(value_parts),
            result=combined,
        )
        return FusionExplanation(ranking_factors=ranking_factors, evidence=evidence, formula=formula)


@dataclass(frozen=True, slots=True)
class RrfFusion:
    """Reciprocal rank fusion: ``sum(1 / (k + rank_s(rowid)))``.

    Ranks are derived from the prepared scores (descending). Because min-max
    and softmax are monotonic, the rank is identical whether computed on raw
    or normalized scores — which is why RRF can skip normalization entirely.
    """

    k: int = 60
    name: str = "rrf"

    @property
    def params(self) -> Mapping[str, object]:
        return {"k": self.k}

    def fuse(self, prepared: Mapping[str, dict[int, float]]) -> dict[int, float]:
        combined: dict[int, float] = {}
        for scores in prepared.values():
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for position, (rowid, _) in enumerate(ranked, start=1):
                combined[rowid] = combined.get(rowid, 0.0) + 1.0 / (self.k + position)
        return combined

    def explain(
        self,
        sources: Mapping[str, SourceExplanation],
        combined: float,
        norm: Norm | None = None,
    ) -> FusionExplanation:
        ranking_factors: dict[str, RankingFactor] = {}
        symbolic_parts: list[str] = []
        value_parts: list[str] = []

        for name, source in sources.items():
            if source.status is not RetrievalStatus.MATCHED:
                continue
            rank = source.pool_rank or 0
            ranking_factors[name] = RankingFactor("rank", rank)
            symbolic_parts.append(f"1 \u00f7 ({self.k} + {name}_rank)")
            value_parts.append(f"1 \u00f7 {self.k + rank}")

        missing = [
            f"{name}: {source.status.label}"
            for name, source in sources.items()
            if source.status is not RetrievalStatus.MATCHED
        ]

        formula = FormulaComponents(
            method_label=f"RRF (k={self.k})",
            symbolic_terms=" + ".join(symbolic_parts),
            value_terms=" + ".join(value_parts),
            result=combined,
            missing_note=", ".join(missing) if missing else None,
        )
        return FusionExplanation(ranking_factors=ranking_factors, evidence={}, formula=formula)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """The retrieval path a query actually runs, frozen at assembly time.

    ``norm`` is ``None`` when the fusion ranks by position (RRF), in which
    case normalization is skipped. ``fusion`` carries its own parameters.
    """

    norm: Norm | None
    fusion: Fusion
