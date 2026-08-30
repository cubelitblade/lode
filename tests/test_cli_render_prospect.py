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

from pathlib import Path, PurePosixPath

from rich.console import Console

from lode.cli.render import RenderOptions
from lode.cli.render.prospect import render_prospect
from lode.index import FileStatus, PathRef
from lode.index.search import ProspectResult, SearchHit


def _hit(*, stale: bool = False) -> SearchHit:
    return SearchHit(
        digest="blake3:0123456789abcdef",
        text="quantum entanglement",
        heading="Intro",
        score=0.75,
        refs=(
            PathRef(
                path=PurePosixPath("docs/report.txt"),
                status=FileStatus.STALE if stale else FileStatus.FRESH,
            ),
        ),
        page=3,
    )


def _result(hits: list[SearchHit], *, has_stale: bool = False) -> ProspectResult:
    return ProspectResult(
        workspace=Path("."),
        query="entanglement",
        top_k=5,
        hits=hits,
        has_stale=has_stale,
    )


def _render(result: ProspectResult, options: RenderOptions | None = None) -> str:
    """Render a prospect report to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_prospect(result, options=options, console=console)
    return console.export_text()


def test_render_prospect_lists_hits_as_cards() -> None:
    text = _render(_result([_hit()]))
    assert "#1 · 0.750" in text
    assert f"{Path('docs') / 'report.txt'} > Intro (p.3)" in text
    assert "quantum entanglement" in text


def test_render_prospect_hides_full_content_address() -> None:
    text = _render(_result([_hit()]))
    assert "blake3:" not in text


def test_render_prospect_marks_stale_hit() -> None:
    text = _render(_result([_hit(stale=True)]))
    assert "[stale]" in text


def test_render_prospect_dry_hole() -> None:
    text = _render(_result([]))
    assert "Dry hole: nothing matched." in text
    assert "quantum entanglement" not in text


def test_render_prospect_warns_stale_in_results() -> None:
    text = _render(_result([_hit(stale=True)], has_stale=True))
    assert "results include stale files" in text


def test_render_prospect_warns_pending_outside_results() -> None:
    text = _render(_result([_hit()], has_stale=True))
    assert "pending changes outside these results" in text


def test_render_prospect_no_warning_when_clean() -> None:
    text = _render(_result([_hit()], has_stale=False))
    assert "Warning" not in text


def test_render_prospect_no_color_keeps_output() -> None:
    text = _render(_result([_hit()]), options=RenderOptions(no_color=True))
    assert "#1 · 0.750" in text
    assert "quantum entanglement" in text
