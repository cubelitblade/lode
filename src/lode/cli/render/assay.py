"""Human-readable `lode assay` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.

``assay`` explains why one chunk scored as it did: it shows the query, the
chunk's source location, and a score table breaking the result down into its
semantic and lexical contributions, the fusion, and the combined score. The
table is data-driven from the ``RetrievalPlan`` so swapping the norm or fusion
algorithm does not require render changes.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lode.cli.render.core import Intent, RenderOptions
from lode.index.ranking import LinearFusion, RetrievalPlan, RrfFusion
from lode.index.search import ScoreExplanation


def render_assay(
    explanation: ScoreExplanation,
    *,
    query: str,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a human-readable score explanation as a table.

    The table has one row per signal (``semantic``, ``lexical``) with columns
    for the normalization, the factor, and the weighted contribution, plus a
    fusion row, followed by the combined ranking score. The whole report is
    wrapped in a bordered panel (matching the other commands); ``console`` may
    be injected (e.g. a recording console) to capture output in tests.
    """
    if options is None:
        options = RenderOptions()
    console = console or Console(no_color=options.no_color)

    chunk = explanation.chunk
    plan = explanation.plan
    info_style = options.intent_colors.get(Intent.INFO, "")

    source = chunk.primary.path
    if len(chunk.refs) > 1:
        source += f" (+{len(chunk.refs) - 1} more)"
    if chunk.heading:
        source += f" > {chunk.heading}"
    if chunk.page is not None:
        source += f" (p.{chunk.page})"
    short_id = chunk.digest.removeprefix("blake3:")[:12]

    table = Table(box=None, show_header=True)
    table.add_column("Signal", style=info_style)
    table.add_column("Normalization", justify="left")
    table.add_column("Factor", justify="left")
    table.add_column("Contribution", justify="left")

    table.add_row("semantic", *_signal_cells(explanation, plan, "semantic"))
    table.add_row("lexical", *_signal_cells(explanation, plan, "lexical"))
    table.add_row("fusion", _fusion_cell(plan), "", "")

    frame = options.box
    if frame is not None:
        body = Group(
            Text(f"Query: {query}", style=info_style),
            "",
            table,
            "",
            Text(f"Ranking score: {explanation.combined:.4f}", style=info_style),
        )
        console.print(
            Panel(
                body,
                title=f"Assay · {short_id}",
                title_align="left",
                border_style=options.border_style,
                box=frame,
            )
        )
    else:
        console.print(f"Query: {query}", style=info_style, markup=False, highlight=False, soft_wrap=True)
        console.print()
        console.print(table)
        console.print()
        console.print(
            f"Ranking score: {explanation.combined:.4f}",
            style=info_style,
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


def _signal_cells(
    explanation: ScoreExplanation,
    plan: RetrievalPlan,
    source: str,
) -> tuple[str, str, str]:
    """The normalization, factor, and contribution cells for one signal.

    When the plan skips normalization (RRF ranks by position), the
    normalization cell explains the skip and factor/contribution are ``n/a``
    (RRF has no per-source weights).
    """
    raw = explanation.semantic_raw if source == "semantic" else explanation.lexical_raw
    prepared = explanation.semantic_prepared if source == "semantic" else explanation.lexical_prepared
    if plan.norm is None:
        return ("skipped (RRF ranks by position)", "n/a", "n/a")
    norm_cell = _normalization(raw, prepared, plan.norm.name)
    if isinstance(plan.fusion, LinearFusion):
        weight = plan.fusion.weights.get(source, 0.0)
        return (norm_cell, _factor(weight), _contribution(prepared, weight))
    return (norm_cell, "n/a", "n/a")


def _fusion_cell(plan: RetrievalPlan) -> str:
    """The fusion row cell: algorithm name plus its parameters."""
    if isinstance(plan.fusion, LinearFusion):
        weights = plan.fusion.weights
        semantic = weights.get("semantic", 0.0)
        lexical = weights.get("lexical", 0.0)
        return f"linear (semantic {semantic}, lexical {lexical})"
    if isinstance(plan.fusion, RrfFusion):
        return f"rrf (k={plan.fusion.k})"
    return plan.fusion.name


def _normalization(raw: float | None, prepared: float | None, norm_name: str) -> str:
    """The normalization cell: ``<norm>: raw -> prepared``, or ``n/a`` when absent."""
    if raw is None or prepared is None:
        return "n/a"
    return f"{norm_name}: {raw:.4f} -> {prepared:.4f}"


def _factor(weight: float) -> str:
    """The factor cell: ``x <weight>`` (multiplication sign)."""
    return f"× {weight}"  # noqa: RUF001 — intentional multiplication-sign glyph


def _contribution(prepared: float | None, weight: float) -> str:
    """The contribution cell: prepared score times its factor."""
    value = (prepared if prepared is not None else 0.0) * weight
    return f"{value:.4f}"
