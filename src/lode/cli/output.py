"""Human-readable rendering for lode CLI output.

Kept separate from ``cli/__init__`` so command wiring stays thin and the
render layer can be unit-tested in isolation. Every render function takes a
``RenderOptions``; defaults reproduce the current rich output so callers that
do not care about styling get the existing behaviour.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lode.cli.render import MARKERS, STATUS_INTENT, Intent, RenderOptions, Status
from lode.ingestion.pipeline import SurveySummary


def render_survey(workspace: Path, result: SurveySummary, options: RenderOptions | None = None) -> None:
    """Render a human-readable survey report.

    De-emphasizes the noisy ``skipped`` count so the actionable signals
    (new / changed / missing) stay scannable. The numbers mirror the JSON
    payload so the two never drift. Symbols are always emitted; colour comes
    from ``options`` and can be switched off via a plain/accessible preset.
    """
    if options is None:
        options = RenderOptions()
    console = Console()

    counts_by_status = {
        Status.NEW: result.new,
        Status.CHANGED: result.changed,
        Status.MISSING: result.missing,
        Status.UNCHANGED: result.unchanged,
        Status.SKIPPED: result.skipped,
    }
    display_order = (Status.NEW, Status.CHANGED, Status.MISSING, Status.UNCHANGED, Status.SKIPPED)
    counts = Text()
    for index, status in enumerate(display_order):
        if index:
            counts.append(" · ")
        style = options.intent_colors.get(STATUS_INTENT[status], "")
        counts.append(f"{MARKERS[status]} {status.value} {counts_by_status[status]}", style=style)

    status_panel = Panel(
        counts,
        title=f"Workspace status ({workspace}):",
        title_align="left",
        border_style=options.border_style,
        box=options.box,  # pyright: ignore[reportArgumentType]  # rich accepts None for no border, but stubs type it as Box
    )
    console.print(status_panel)

    if result.pending:
        pending = Table(box=None, show_header=False)
        pending.add_column("Change", width=1, justify="center")
        pending.add_column("Path")
        for status, paths in (
            (Status.NEW, result.new_files),
            (Status.CHANGED, result.changed_files),
            (Status.MISSING, result.missing_files),
        ):
            style = options.intent_colors.get(STATUS_INTENT[status], "")
            for path in paths:
                pending.add_row(Text(MARKERS[status], style=style), Text(path, style=style))
        panel = Panel(
            pending,
            title=f"Pending sync ({result.pending} files)",
            title_align="left",
            border_style=options.border_style,
            box=options.box,  # pyright: ignore[reportArgumentType]  # rich accepts None for no border, but stubs type it as Box
        )
        console.print()
        console.print(panel)
        console.print()
        console.print(
            " Run `lode mine` to update index.",
            style=options.intent_colors.get(Intent.INFO, ""),
        )
