"""Score explanation data structures.

Pure data types consumed by the ``assay why`` render layer. The ``Fusion``
protocol returns a ``FusionExplanation`` from its ``explain()`` method; the
render layer formats it without branching on concrete operator types.

Migrated from ``search.py``: ``RetrievalStatus``, ``SourceExplanation``,
``ScoreExplanation``. New: ``RankingFactor``, ``EvidenceBlock``,
``FormulaComponents``, ``FusionExplanation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lode.index.ranking import RetrievalPlan
    from lode.index.store import ChunkWithPath


class RetrievalStatus(Enum):
    """One retrieval source's participation in a query.

    The four states are mutually exclusive and derivable from the scoring
    pass: ``disabled`` (never queried — e.g. a zero linear weight), ``empty``
    (queried but returned no candidates at all), ``not_retrieved`` (the pool
    has candidates but not this chunk), ``matched`` (this chunk is in the
    pool with scores).
    """

    DISABLED = "disabled"
    EMPTY = "empty"
    NOT_RETRIEVED = "not_retrieved"
    MATCHED = "matched"

    @property
    def label(self) -> str:
        """Human-readable label for this status."""
        return {
            RetrievalStatus.DISABLED: "disabled",
            RetrievalStatus.EMPTY: "no results",
            RetrievalStatus.NOT_RETRIEVED: "not retrieved",
            RetrievalStatus.MATCHED: "matched",
        }[self]


@dataclass(frozen=True, slots=True)
class SourceExplanation:
    """Per-source facts for one chunk in one query.

    Invariant: when ``status`` is ``MATCHED``, ``raw_score``, ``prepared_score``
    and ``pool_rank`` are all non-None; for every other status they are all
    None. ``pool_size`` is the source's actual candidate count (0 unless the
    source was queried).
    """

    status: RetrievalStatus
    pool_size: int
    raw_score: float | None = None
    prepared_score: float | None = None
    pool_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RankingFactor:
    """One ranking factor for one source in the ``FusionExplanation``.

    The source identity is the mapping key in ``FusionExplanation.ranking_factors``,
    not a field here — the key is the single source of truth.
    """

    metric: str
    """What this factor measures: ``"contribution"`` (weighted score) or ``"rank"``."""

    value: float
    """The numeric value to display."""


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    """Per-source evidence detail for one MATCHED source.

    All fields are optional: RRF fills only ``rank`` and ``raw_score``
    (no normalization, no weight/contribution). Linear fills all fields.
    The render layer decides formatting based on which fields are non-None.
    """

    rank: int | None = None
    raw_score: float | None = None
    prepared_score: float | None = None
    normalization: str | None = None
    weight: float | None = None
    contribution: float | None = None


@dataclass(frozen=True, slots=True)
class FormulaComponents:
    """Self-contained fusion formula display data.

    The render layer formats this into the Fusion section without any
    isinstance checks on concrete fusion types.
    """

    method_label: str
    """e.g. ``"RRF (k=60)"`` or ``"Linear"``."""

    symbolic_terms: str
    """e.g. ``"1 ÷ (60 + semantic_rank)"`` or ``"semantic × 0.7 + lexical × 0.3"``."""  # noqa: RUF001

    value_terms: str
    """e.g. ``"1 ÷ 61 + 1 ÷ 63"`` or ``"0.9000 × 0.7 + 0.5000 × 0.3"``."""  # noqa: RUF001

    result: float
    """The combined score after fusion."""

    missing_note: str | None = None
    """Optional note for non-matched sources, e.g. ``"lexical: no results"``."""


@dataclass(frozen=True, slots=True)
class FusionExplanation:
    """Complete render-ready explanation from ``Fusion.explain()``.

    Consumed by the ``assay why`` render layer. The render layer formats
    this without branching on concrete operator types — all fusion-specific
    logic lives in the ``Fusion.explain()`` implementation.
    """

    ranking_factors: Mapping[str, RankingFactor]
    """One entry per MATCHED source. Key is the source name (e.g. ``"semantic"``)."""

    evidence: Mapping[str, EvidenceBlock]
    """One entry per MATCHED source. Empty for RRF (no normalization/weight info)."""

    formula: FormulaComponents
    """The fusion formula with actual numbers substituted."""


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    """Why one chunk scored as it did for a query.

    ``sources`` carries the per-source facts (status, pool, scores) keyed by
    source name; ``combined`` is the fusion output and ``rank`` the chunk's
    true rank across all scored rows (not truncated to ``top_k``), with
    ``in_results`` reporting whether it made the returned top-``top_k``.
    ``plan`` is the retrieval path that produced the scores (the flow context
    ``assay`` reads to explain them).
    """

    chunk: ChunkWithPath
    sources: Mapping[str, SourceExplanation]
    combined: float
    rank: int | None
    in_results: bool
    plan: RetrievalPlan
    top_k: int
