"""Human-readable `lode assay` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.

``assay`` explains why one chunk scored as it did: it shows the query, the
chunk's source location, and a score table breaking the result down into its
semantic and lexical contributions and the combined score.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lode.cli.render.core import Intent, RenderOptions
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
    for the min-max normalization, the factor, and the weighted contribution,
    followed by the combined ranking score. The whole report is wrapped in a
    bordered panel (matching the other commands); ``console`` may be injected
    (e.g. a recording console) to capture output in tests.
    """
    if options is None:
        options = RenderOptions()
    console = console or Console(no_color=options.no_color)

    chunk = explanation.chunk
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

    table.add_row(
        "semantic",
        _normalization(explanation.semantic_raw, explanation.semantic_norm),
        _factor(explanation.semantic_weight),
        _contribution(explanation.semantic_norm, explanation.semantic_weight),
    )
    table.add_row(
        "lexical",
        _normalization(explanation.lexical_raw, explanation.lexical_norm),
        _factor(explanation.lexical_weight),
        _contribution(explanation.lexical_norm, explanation.lexical_weight),
    )

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


def _normalization(raw: float | None, norm: float | None) -> str:
    """The min-max normalization cell: ``raw -> norm``, or ``n/a`` when absent."""
    if raw is None or norm is None:
        return "n/a"
    return f"min-max: {raw:.4f} -> {norm:.4f}"


def _factor(weight: float) -> str:
    """The factor cell: ``x <weight>`` (multiplication sign)."""
    return f"× {weight}"  # noqa: RUF001 — intentional multiplication-sign glyph


def _contribution(norm: float | None, weight: float) -> str:
    """The contribution cell: normalized score times its factor."""
    value = (norm if norm is not None else 0.0) * weight
    return f"{value:.4f}"
