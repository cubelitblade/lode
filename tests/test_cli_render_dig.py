"""Unit tests for `lode.cli.render.dig.render_dig`.

Human-readable CLI output is presentation, not a stable contract, so these
tests assert behavioural/structural signals rather than exact text or spacing:

* the window context header is reported;
* each chunk is a card with its source and full text;
* the center chunk is marked;
* a stale chunk is flagged.

Colour is deliberately not asserted here — it is stripped in non-TTY test runs
and is an intent/palette concern covered by `lode.cli.render`.
"""

from __future__ import annotations

from rich.console import Console

from lode.cli.render import RenderOptions, render_options_from_preset
from lode.cli.render.dig import render_dig
from lode.index.store import ChunkWithPath, FileStatus


def _chunk(seq: int, *, stale: bool = False, text: str = "full chunk text") -> ChunkWithPath:
    return ChunkWithPath(
        chunk_id="blake3:0123456789abcdef",
        text=text,
        heading="Intro",
        path="docs/report.txt",
        file_status=FileStatus.STALE if stale else FileStatus.CURRENT,
        page=3,
        seq=seq,
    )


def _render(
    chunks: list[ChunkWithPath],
    *,
    digest: str = "0123456789ab",
    center_seq: int | None,
    radius: int,
    options: RenderOptions | None = None,
) -> str:
    """Render a dig window to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_dig(chunks, digest=digest, center_seq=center_seq, radius=radius, options=options, console=console)
    return console.export_text()


def test_render_dig_reports_dug_digest_with_radius() -> None:
    text = _render([_chunk(5)], center_seq=5, radius=1)
    assert "Dug 0123456789ab with radius 1" in text
    assert "Window" not in text


def test_render_dig_reports_dug_digest_without_radius() -> None:
    text = _render([_chunk(5)], center_seq=5, radius=0)
    assert "Dug 0123456789ab" in text
    assert "with radius" not in text


def test_render_dig_marks_center_chunk() -> None:
    text = _render([_chunk(5)], center_seq=5, radius=0)
    assert "5 · center" in text
    assert "docs/report.txt > Intro (p.3)" in text
    assert "full chunk text" in text


def test_render_dig_lists_neighbor_seq() -> None:
    text = _render([_chunk(4), _chunk(5)], center_seq=5, radius=1)
    assert "4" in text
    assert "5 · center" in text


def test_render_dig_marks_stale_chunk() -> None:
    text = _render([_chunk(5, stale=True)], center_seq=5, radius=1)
    assert "[stale]" in text


def test_render_dig_hides_full_content_address() -> None:
    text = _render([_chunk(5)], center_seq=5, radius=0)
    assert "blake3:" not in text


def test_render_dig_plain_preset_keeps_output() -> None:
    text = _render([_chunk(5)], center_seq=5, radius=0, options=render_options_from_preset("plain"))
    assert "5 · center" in text
    assert "full chunk text" in text
