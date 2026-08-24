"""Human-readable `lode prospect` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.

``prospect`` renders hits as a stack of cards: each one is a bordered panel
titled ``#<rank> · <score>``, showing the source location (path, heading, page,
stale tag), a preview of the chunk, and a muted digest so ``dig`` can follow it.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from lode.cli.render.core import Intent, RenderOptions
from lode.cli.render.output import preview
from lode.index.search import SearchHit
from lode.index.store import FileStatus


def render_prospect(
    workspace: Path,
    query: str,
    hits: list[SearchHit],
    *,
    stale_warning: str | None = None,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a human-readable prospect report as a stack of cards.

    Each hit is a card titled ``#<rank> · <score>``: the source location
    (``path > heading (p.N) [stale]``), an indented preview snippet, and a
    muted chunk digest so ``lode dig`` can follow it. An empty result prints a
    single ``Dry hole`` line. ``workspace``/``query`` are carried for a future
    header but are not currently displayed.

    ``stale_warning`` is a precomputed message (from the caller's store scan);
    when set it is emitted after the cards with ``WARNING`` intent. The border
    follows ``options.border``; ``console`` may be injected (e.g. a recording
    console) to capture output in tests.
    """
    console = console or Console()
    if options is None:
        options = RenderOptions()

    if not hits:
        console.print(
            "Dry hole: nothing matched.",
            style=options.intent_colors.get(Intent.INFO, ""),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        return

    frame = options.box
    stale_style = options.intent_colors.get(Intent.WARNING, "")
    muted_style = options.intent_colors.get(Intent.MUTED, "")
    for index, hit in enumerate(hits, start=1):
        title = f"#{index} · {hit.score:.3f}"
        source = hit.primary.path
        if len(hit.refs) > 1:
            source += f" (+{len(hit.refs) - 1} more)"
        if hit.heading:
            source += f" > {hit.heading}"
        if hit.page is not None:
            source += f" (p.{hit.page})"
        short_id = hit.digest.removeprefix("blake3:")[:12]
        primary_stale = hit.primary.status is FileStatus.STALE

        if frame is not None:
            body = Text()
            body.append(source)
            if primary_stale:
                body.append(" [stale]", style=stale_style)
            body.append("\n\n")
            body.append(preview(hit.text))
            body.append(f"\n\n{short_id}", style=muted_style)
            console.print(
                Panel(
                    body,
                    title=title,
                    title_align="left",
                    border_style=options.border_style,
                    box=frame,
                )
            )
        else:
            console.print(title)
            source_line = f"  {source}"
            if primary_stale:
                source_line += " [stale]"
            console.print(source_line, style=stale_style if primary_stale else "")
            console.print(f"  {preview(hit.text)}")
            console.print(f"  {short_id}", style=muted_style)
        console.print()

    if stale_warning:
        console.print(
            stale_warning,
            style=options.intent_colors.get(Intent.WARNING, ""),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
