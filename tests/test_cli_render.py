"""Unit tests for `lode.cli.render.survey.render_survey`.

Human-readable CLI output is presentation, not a stable contract, so these
tests assert behavioural/structural signals rather than exact text or spacing:

* pending paths are reported;
* the status symbols are always emitted (independent of palette);
* border is a separate axis: the default draws a frame, ``Border.NONE`` does not.

Colour is deliberately not asserted here — it is stripped in non-TTY test runs
and is an intent/palette concern covered by `lode.cli.render`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from lode.cli.commands._common import resolve_render_options
from lode.cli.render import (
    ACCESSIBLE_DARK_INTENT_COLORS,
    ACCESSIBLE_LIGHT_INTENT_COLORS,
    ANSI_INTENT_COLORS,
    Border,
    RenderOptions,
    render_options_from_preset,
)
from lode.cli.render.survey import render_survey
from lode.ingestion.pipeline import DetectResult


def _render(result: DetectResult, options: RenderOptions | None = None) -> str:
    """Render a survey report to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    render_survey(Path("."), result, options=options, console=console)
    return console.export_text()


def test_render_survey_lists_pending_paths() -> None:
    result = DetectResult.from_paths(
        new_files=["docs/intro.md"],
        changed_files=["README.md"],
        missing_files=["old.txt"],
        skipped=3,
    )
    text = _render(result)
    assert str(Path("docs") / "intro.md") in text
    assert "README.md" in text
    assert "old.txt" in text


def test_render_survey_lists_renamed_pair() -> None:
    result = DetectResult.from_paths(renamed_files=[("old.txt", "new.txt")])
    text = _render(result)
    assert "old.txt -> new.txt" in text


def test_render_survey_always_emits_symbols() -> None:
    result = DetectResult.from_paths(new_files=["a.md"], changed_files=["b.md"], missing_files=["c.md"], skipped=1)
    text = _render(result)
    assert "+" in text
    assert "~" in text
    assert "-" in text


def test_render_survey_rich_default_has_border() -> None:
    text = _render(DetectResult.from_paths(new_files=["a.md"], skipped=1))
    assert "╭" in text
    assert "╯" in text


def test_render_survey_borderless_has_no_frame() -> None:
    text = _render(DetectResult.from_paths(new_files=["a.md"], skipped=1), RenderOptions(border=Border.NONE))
    assert "╭" not in text
    assert "│" not in text
    assert "a.md" in text


def test_render_survey_no_color_keeps_symbols() -> None:
    text = _render(
        DetectResult.from_paths(new_files=["a.md"], skipped=1),
        RenderOptions(no_color=True),
    )
    assert "+" in text
    assert "a.md" in text


def test_render_preset_ansi_is_default() -> None:
    assert render_options_from_preset("ansi").intent_colors == ANSI_INTENT_COLORS


def test_render_preset_accessible_light() -> None:
    assert render_options_from_preset("accessible_light").intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS


def test_render_preset_accessible_dark() -> None:
    assert render_options_from_preset("accessible_dark").intent_colors == ACCESSIBLE_DARK_INTENT_COLORS


def test_render_preset_plain_and_rich_rejected() -> None:
    with pytest.raises(ValueError):
        render_options_from_preset("plain")
    with pytest.raises(ValueError):
        render_options_from_preset("rich")


def test_resolve_render_options_no_color_flag_wins() -> None:
    """--no-color (flag) forces colour off regardless of palette."""
    options = resolve_render_options(
        configured_palette="ansi",
        configured_no_color=None,
        palette="accessible_light",
        no_color=True,
    )
    assert options.no_color is True
    assert options.intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS


def test_resolve_render_options_config_no_color_wins() -> None:
    """Configured no_color forces colour off even when --palette is passed."""
    options = resolve_render_options(
        configured_palette="ansi",
        configured_no_color=True,
        palette="accessible_light",
        no_color=False,
    )
    assert options.no_color is True
    assert options.intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS


def test_resolve_render_options_config_no_color_false_forces_color() -> None:
    """An explicit configured no_color=false keeps colour on (overrides NO_COLOR)."""
    options = resolve_render_options(
        configured_palette="ansi",
        configured_no_color=False,
    )
    assert options.no_color is False
    assert options.intent_colors == ANSI_INTENT_COLORS


def test_resolve_render_options_palette_flag_overrides_config() -> None:
    """--palette overrides the configured palette when colour is on."""
    options = resolve_render_options(
        configured_palette="ansi",
        configured_no_color=None,
        palette="accessible_light",
        no_color=False,
    )
    assert options.no_color is None
    assert options.intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS


def test_resolve_render_options_defaults_to_config() -> None:
    """No flags: the configured palette is used and no_color is unset."""
    options = resolve_render_options(
        configured_palette="accessible_light",
        configured_no_color=None,
    )
    assert options.no_color is None
    assert options.intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS
