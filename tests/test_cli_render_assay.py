"""Unit tests for `lode.cli.render.assay.render_assay`.

Human-readable CLI output is presentation, not a stable contract, so these
tests assert behavioural/structural signals rather than exact text or spacing:

* the query is shown;
* the score table shows per-source normalization, factor, and contribution;
* the combined ranking score is shown as a footer;
* a disabled source (weight 0) and an absent source (n/a) are handled.

Colour is deliberately not asserted here — it is stripped in non-TTY test runs
and is an intent/palette concern covered by `lode.cli.render`.
"""

from __future__ import annotations

from rich.console import Console

from lode.cli.render import RenderOptions
from lode.cli.render.assay import render_assay
from lode.index.search import ScoreExplanation
from lode.index.store import ChunkWithPath, FileStatus, PathRef


def _explanation(
    *,
    semantic_raw: float | None = 0.9,
    lexical_raw: float | None = -5.0,
    semantic_norm: float | None = 1.0,
    lexical_norm: float | None = 0.5,
    combined: float = 0.8,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> ScoreExplanation:
    return ScoreExplanation(
        chunk=ChunkWithPath(
            digest="blake3:0123456789abcdef",
            text="quantum entanglement",
            heading="Intro",
            refs=(PathRef(path="docs/report.txt", status=FileStatus.FRESH),),
            page=3,
            seq=2,
        ),
        semantic_raw=semantic_raw,
        lexical_raw=lexical_raw,
        semantic_norm=semantic_norm,
        lexical_norm=lexical_norm,
        semantic_pool_rank=1,
        lexical_pool_rank=3,
        semantic_pool_size=40,
        lexical_pool_size=10,
        combined=combined,
        rank=1,
        in_results=True,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        top_k=5,
    )


def _render(explanation: ScoreExplanation, options: RenderOptions | None = None) -> str:
    """Render an explanation to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_assay(explanation, query="entanglement", options=options, console=console)
    return console.export_text()


def test_render_assay_shows_query() -> None:
    text = _render(_explanation())
    assert "Query: entanglement" in text


def test_render_assay_shows_signal_rows() -> None:
    text = _render(_explanation())
    assert "semantic" in text
    assert "lexical" in text


def test_render_assay_shows_normalization() -> None:
    text = _render(_explanation())
    assert "min-max: 0.9000 -> 1.0000" in text
    assert "min-max: -5.0000 -> 0.5000" in text


def test_render_assay_shows_factor_and_contribution() -> None:
    text = _render(_explanation())
    # semantic: 1.0 × 0.7 = 0.7000; lexical: 0.5 × 0.3 = 0.1500  # noqa: RUF003 — intentional multiplication-sign glyph
    assert "× 0.7" in text  # noqa: RUF001 — intentional multiplication-sign glyph
    assert "× 0.3" in text  # noqa: RUF001 — intentional multiplication-sign glyph
    assert "0.7000" in text
    assert "0.1500" in text


def test_render_assay_shows_ranking_score() -> None:
    text = _render(_explanation())
    assert "Ranking score: 0.8000" in text


def test_render_assay_hides_full_content_address() -> None:
    text = _render(_explanation())
    assert "blake3:" not in text


def test_render_assay_marks_absent_source() -> None:
    text = _render(_explanation(lexical_raw=None, lexical_norm=None))
    assert "n/a" in text


def test_render_assay_disabled_source_contributes_zero() -> None:
    text = _render(_explanation(semantic_raw=None, semantic_norm=None, semantic_weight=0.0))
    assert "n/a" in text
    assert "0.0000" in text


def test_render_assay_no_color_keeps_output() -> None:
    text = _render(_explanation(), options=RenderOptions(no_color=True))
    assert "Ranking score: 0.8000" in text
    assert "semantic" in text
