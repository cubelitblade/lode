"""Human-readable `lode survey` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lode.cli.render.core import MARKERS, STATUS_INTENT, Intent, RenderOptions, Status
from lode.ingestion.pipeline import SurveySummary


def render_survey(
    workspace: Path,
    result: SurveySummary,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a human-readable survey report.

    De-emphasizes the noisy ``skipped`` count so the actionable signals
    (new / changed / missing) stay scannable. The numbers mirror the JSON
    payload so the two never drift. Symbols are always emitted; colour comes
    from ``options`` and is independent of the border (``options.border``).

    ``console`` may be injected (e.g. a recording console) to capture output
    in tests; when omitted a default console is used.
    """
    console = console or Console()
    if options is None:
        options = RenderOptions()

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

    frame = options.box
    if frame is not None:
        status_panel = Panel(
            counts,
            title=f"Workspace status ({workspace}):",
            title_align="left",
            border_style=options.border_style,
            box=frame,
        )
        console.print(status_panel)
    else:
        console.print(f"Workspace status ({workspace}):")
        indented_counts = Text("  ")
        indented_counts.append_text(counts)
        console.print(indented_counts)

    if result.pending:
        header = f"Pending sync ({result.pending} files)"
        if frame is not None:
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
            console.print()
            console.print(
                Panel(pending, title=header, title_align="left", border_style=options.border_style, box=frame)
            )
        else:
            console.print()
            console.print(header)
            for status, paths in (
                (Status.NEW, result.new_files),
                (Status.CHANGED, result.changed_files),
                (Status.MISSING, result.missing_files),
            ):
                style = options.intent_colors.get(STATUS_INTENT[status], "")
                for path in paths:
                    console.print(f"  {MARKERS[status]} {path}", style=style)
        hint_indent = " " if frame is not None else "  "
        console.print()
        console.print(
            f"{hint_indent}Run `lode mine` to update index.",
            style=options.intent_colors.get(Intent.INFO, ""),
        )
