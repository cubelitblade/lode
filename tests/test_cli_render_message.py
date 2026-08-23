"""Unit tests for `lode.cli.render.output.render_message`.

``render_message`` emits a single human-readable line with an ``Intent``-based
colour. Colour is stripped in non-TTY test runs, so these tests assert
behavioural signals (message present, multi-line preserved, plain preset still
emits the text) rather than exact ANSI output.
"""

from __future__ import annotations

from rich.console import Console

from lode.cli.render import Intent, RenderOptions, render_options_from_preset
from lode.cli.render.output import render_message


def _render(message: str, *, intent: Intent, options: RenderOptions | None = None) -> str:
    """Render a single message to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_message(message, intent=intent, options=options, console=console)
    return console.export_text()


def test_render_message_emits_text() -> None:
    text = _render("The current lode was driven at 512 dimensions.", intent=Intent.ERROR)
    assert "The current lode was driven at 512 dimensions." in text


def test_render_message_keeps_multi_line() -> None:
    message = (
        "The current lode was driven at 512 dimensions, but this model yields "
        "1024-wide nuggets — they don't sit on the same vein.\n"
        "Hint: Re-mine it with `lode mine --from-scratch`, or switch your embedding config "
        "back to the model/dimension that dug it."
    )
    text = _render(message, intent=Intent.ERROR)
    assert "they don't sit on the same vein." in text
    assert "Hint: Re-mine it with `lode mine --from-scratch`" in text


def test_render_message_plain_preset_keeps_text() -> None:
    text = _render("something went wrong", intent=Intent.ERROR, options=render_options_from_preset("plain"))
    assert "something went wrong" in text


def test_render_message_does_not_highlight_numbers() -> None:
    """rich's default highlighting re-colours tokens such as numbers on top of
    the ``intent`` style; ``render_message`` must keep the message verbatim."""
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor")
    render_message("driven at 512 dimensions", intent=Intent.ERROR, console=console)
    out = buf.getvalue()
    # Default rich number highlight is bold-cyan (1;36); the base ERROR style is red (31).
    assert "1;36" not in out
    assert "\x1b[31m" in out
