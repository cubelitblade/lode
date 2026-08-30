"""Unit tests for `lode.cli.render.mine.render_mine`.

Human-readable CLI output is presentation, not a stable contract, so these
tests assert behavioural/structural signals rather than exact text or spacing:

* the status counts are reported;
* the actionable paths (added / updated / removed) are listed;
* the status symbols are always emitted (independent of palette);
* failures are surfaced with a hint to re-mine.

Colour is deliberately not asserted here — it is stripped in non-TTY test runs
and is an intent/palette concern covered by `lode.cli.render`.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from rich.console import Console

from lode.cli.render import RenderOptions
from lode.cli.render.mine import render_mine
from lode.ingestion.pipeline import FailedFile, SyncSummary


def _render(result: SyncSummary, options: RenderOptions | None = None) -> str:
    """Render a mine report to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_mine(Path("."), result, options=options, console=console)
    return console.export_text()


def test_render_mine_reports_counts() -> None:
    result = SyncSummary(
        added_files=[PurePosixPath("a.md")],
        updated_files=[PurePosixPath("b.md")],
        removed_files=[PurePosixPath("c.md")],
        unchanged=2,
        skipped=1,
    )
    text = _render(result)
    # Normalize whitespace: panel wrapping may split a "marker label count"
    # group across lines, but the tokens stay adjacent in reading order.
    flat = " ".join(text.split())
    assert "+ added 1" in flat
    assert "~ updated 1" in flat
    assert "- removed 1" in flat
    assert "= unchanged 2" in flat
    assert "○ skipped 1" in flat


def test_render_mine_reports_renamed() -> None:
    result = SyncSummary(renamed_files=[(PurePosixPath("old.txt"), PurePosixPath("new.txt"))])
    text = _render(result)
    assert "old.txt -> new.txt" in text


def test_render_mine_lists_changed_paths() -> None:
    result = SyncSummary(
        added_files=[PurePosixPath("docs/intro.md")],
        updated_files=[PurePosixPath("README.md")],
        removed_files=[PurePosixPath("old.txt")],
    )
    text = _render(result)
    assert str(Path("docs") / "intro.md") in text
    assert "README.md" in text
    assert "old.txt" in text


def test_render_mine_always_emits_symbols() -> None:
    result = SyncSummary(
        added_files=[PurePosixPath("a.md")],
        updated_files=[PurePosixPath("b.md")],
        removed_files=[PurePosixPath("c.md")],
    )
    text = _render(result)
    assert "+" in text
    assert "~" in text
    assert "-" in text


def test_render_mine_reports_failures() -> None:
    result = SyncSummary(
        added_files=[],
        failed=[FailedFile(path=PurePosixPath("bad.docx"), error="unsupported format")],
    )
    text = _render(result)
    assert "Stumbled on" in text
    assert "× failed 1" in text  # noqa: RUF001 — multiplication-sign glyph is the intentional error marker
    assert "×" in text  # noqa: RUF001 — multiplication-sign glyph is the intentional error marker
    assert "bad.docx" in text
    assert "unsupported format" in text
    assert "Re-run `lode mine` after fixing these to retry." in text


def test_render_mine_no_color_keeps_symbols() -> None:
    result = SyncSummary(added_files=[PurePosixPath("a.md")], skipped=1)
    text = _render(result, RenderOptions(no_color=True))
    assert "+" in text
    assert "a.md" in text


def test_render_mine_empty_shows_nothing_to_do() -> None:
    result = SyncSummary(unchanged=3, skipped=2)
    text = _render(result)
    assert "Nothing to do." in text
