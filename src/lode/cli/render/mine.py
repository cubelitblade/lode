"""Human-readable `lode mine` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.

``mine`` mirrors the visual language of ``survey``: a bordered summary panel
plus a table of the changed files. When there is nothing to do, a single
``Nothing to do.`` line is shown instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lode.cli.render.core import MARKERS, STATUS_INTENT, Entry, Intent, RenderOptions, Status, entry_label
from lode.ingestion.pipeline import SyncSummary


def _processed_entries(result: SyncSummary) -> Sequence[tuple[Status, Sequence[Entry]]]:
    """Processed work grouped by status, in display order."""
    return [
        (Status.NEW, result.added_files),
        (Status.CHANGED, result.updated_files),
        (Status.RENAMED, result.renamed_files),
        (Status.MISSING, result.removed_files),
    ]


def render_mine(
    workspace: Path,
    result: SyncSummary,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a human-readable mine report.

    Mirrors the JSON payload so the numbers never drift. The actionable lists
    (``added``/``updated``/``removed``) always emit their path markers
    (``+``/``~``/``-``), independent of palette; ``unchanged``/``skipped`` are
    count-only. Colour comes from ``options.intent_colors``; ``console`` may be
    injected (e.g. a recording console) to capture output in tests.
    """
    if options is None:
        options = RenderOptions()
    console = console or Console(no_color=options.no_color)

    if result.added == 0 and result.updated == 0 and result.removed == 0 and result.renamed == 0 and not result.failed:
        console.print("Nothing to do.", style=options.intent_colors.get(Intent.INFO, ""))
        return

    counts = Text()
    statuses: list[tuple[Status, str, int]] = [
        (Status.NEW, "added", result.added),
        (Status.CHANGED, "updated", result.updated),
        (Status.MISSING, "removed", result.removed),
        (Status.UNCHANGED, "unchanged", result.unchanged),
        (Status.SKIPPED, "skipped", result.skipped),
    ]
    if result.failed:
        # Show the failed count only when non-zero, so the summary line does
        # not grow with a no-op "failed 0" (and stay within the panel width).
        statuses.insert(3, (Status.FAILED, "failed", len(result.failed)))

    for index, (status, label, count) in enumerate(statuses):
        if index:
            counts.append(" · ")
        style = options.intent_colors.get(STATUS_INTENT[status], "")
        counts.append(f"{MARKERS[status]} {label} {count}", style=style)

    frame = options.box
    if frame is not None:
        counts_panel = Panel(
            counts,
            title=f"Mining completed ({workspace})",
            title_align="left",
            border_style=options.border_style,
            box=frame,
        )
        console.print(counts_panel)
    else:
        console.print(f"Mining completed ({workspace})")
        indented_counts = Text("  ")
        indented_counts.append_text(counts)
        console.print(indented_counts)

    if result.added or result.updated or result.removed or result.renamed:
        header = f"Processed files ({result.added + result.updated + result.removed + result.renamed})"
        if frame is not None:
            changed = Table(box=None, show_header=False)
            changed.add_column("Change", width=1, justify="center")
            changed.add_column("Path")
            for status, entries in _processed_entries(result):
                style = options.intent_colors.get(STATUS_INTENT[status], "")
                for entry in entries:
                    changed.add_row(
                        Text(MARKERS[status], style=style),
                        Text(entry_label(entry), style=style),
                    )
            console.print()
            console.print(
                Panel(changed, title=header, title_align="left", border_style=options.border_style, box=frame)
            )
        else:
            console.print()
            console.print(header)
            for status, entries in _processed_entries(result):
                style = options.intent_colors.get(STATUS_INTENT[status], "")
                for entry in entries:
                    console.print(f"  {MARKERS[status]} {entry_label(entry)}", style=style)

    if result.failed:
        error_style = options.intent_colors.get(STATUS_INTENT[Status.FAILED], "")
        failed_marker = MARKERS[Status.FAILED]
        if frame is not None:
            failed_text = Text()
            for index, failure in enumerate(result.failed):
                if index:
                    failed_text.append("\n")
                failed_text.append(f"{failed_marker} {failure.path}\n", style=error_style)
                failed_text.append(f"  {failure.error}", style=error_style)
            console.print()
            console.print(
                Panel(
                    failed_text,
                    title="Stumbled on",
                    title_align="left",
                    border_style=options.border_style,
                    box=frame,
                )
            )
        else:
            console.print()
            console.print("Stumbled on", style=error_style)
            for failure in result.failed:
                console.print(f"{failed_marker} {failure.path}", style=error_style)
                console.print(f"  {failure.error}", style=error_style)
        hint_indent = " " if frame is not None else "  "
        console.print()
        console.print(
            f"{hint_indent}Re-run `lode mine` after fixing these to retry.",
            style=options.intent_colors.get(Intent.INFO, ""),
        )
