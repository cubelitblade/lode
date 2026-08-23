"""Human-readable `lode dig` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.

``dig`` renders the chunk window as a stack of cards: each is a bordered panel
titled ``<seq>`` (with ``· center`` for the target chunk), showing the source
location (path, heading, page, stale tag), the full chunk text, and a muted
digest.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from lode.cli.render.core import Intent, RenderOptions
from lode.index.store import ChunkWithPath, FileStatus


def render_dig(
    chunks: list[ChunkWithPath],
    *,
    digest: str,
    center_seq: int | None,
    radius: int,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a human-readable dig window as a stack of cards.

    Each chunk is a card titled ``<seq>`` (``· center`` for the target),
    containing the source location (``path > heading (p.N) [stale]``), the full
    chunk text, and a muted digest. The header line reports the dug digest (and
    the radius when one was requested). The border follows ``options.border``;
    ``console`` may be injected (e.g. a recording console) to capture output in
    tests.
    """
    console = console or Console()
    if options is None:
        options = RenderOptions()

    header = f"Dug {digest}" if radius == 0 else f"Dug {digest} with radius {radius}."
    console.print(
        header,
        style=options.intent_colors.get(Intent.INFO, ""),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )

    frame = options.box
    stale_style = options.intent_colors.get(Intent.WARNING, "")
    muted_style = options.intent_colors.get(Intent.MUTED, "")
    for chunk in chunks:
        source = chunk.path
        if chunk.heading:
            source += f" > {chunk.heading}"
        if chunk.page is not None:
            source += f" (p.{chunk.page})"
        short_id = chunk.chunk_id.removeprefix("blake3:")[:12]
        seq_label = chunk.seq if chunk.seq is not None else short_id
        title = str(seq_label)
        if chunk.seq == center_seq:
            title += " · center"
        is_stale = chunk.file_status is FileStatus.STALE

        if frame is not None:
            body = Text()
            body.append(source)
            if is_stale:
                body.append(" [stale]", style=stale_style)
            body.append("\n\n")
            body.append(chunk.text)
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
            if is_stale:
                source_line += " [stale]"
            console.print(source_line, style=stale_style if is_stale else "")
            console.print(f"  {chunk.text}")
            console.print(f"  {short_id}", style=muted_style)
        console.print()
