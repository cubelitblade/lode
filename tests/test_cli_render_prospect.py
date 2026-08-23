"""Unit tests for `lode.cli.render.prospect.render_prospect`.

Human-readable CLI output is presentation, not a stable contract, so these
tests assert behavioural/structural signals rather than exact text or spacing:

* each hit is a card titled ``#<rank> · <score>`` with its source and preview;
* the full content-address prefix is hidden (only a short id is shown);
* an empty result prints a single ``Dry hole`` line;
* a stale warning is emitted when supplied.

Colour is deliberately not asserted here — it is stripped in non-TTY test runs
and is an intent/palette concern covered by `lode.cli.render`.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from lode.cli.render import RenderOptions, render_options_from_preset
from lode.cli.render.prospect import render_prospect
from lode.index.search import SearchHit


def _hit(*, stale: bool = False) -> SearchHit:
    return SearchHit(
        chunk_id="blake3:0123456789abcdef",
        text="quantum entanglement",
        path="docs/report.txt",
        heading="Intro",
        score=0.75,
        stale=stale,
        page=3,
    )


def _render(hits: list[SearchHit], *, stale_warning: str | None = None, options: RenderOptions | None = None) -> str:
    """Render a prospect report to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_prospect(Path("."), "entanglement", hits, stale_warning=stale_warning, options=options, console=console)
    return console.export_text()


def test_render_prospect_lists_hits_as_cards() -> None:
    text = _render([_hit()])
    assert "#1 · 0.750" in text
    assert "docs/report.txt > Intro (p.3)" in text
    assert "quantum entanglement" in text


def test_render_prospect_hides_full_content_address() -> None:
    text = _render([_hit()])
    assert "blake3:" not in text


def test_render_prospect_marks_stale_hit() -> None:
    text = _render([_hit(stale=True)])
    assert "[stale]" in text


def test_render_prospect_dry_hole() -> None:
    text = _render([])
    assert "Dry hole: nothing matched." in text
    assert "quantum entanglement" not in text


def test_render_prospect_emits_stale_warning() -> None:
    text = _render([_hit()], stale_warning="Warning: results include stale files; verify them before relying on them.")
    assert "results include stale files" in text


def test_render_prospect_plain_preset_keeps_output() -> None:
    text = _render([_hit()], options=render_options_from_preset("plain"))
    assert "#1 · 0.750" in text
    assert "quantum entanglement" in text
