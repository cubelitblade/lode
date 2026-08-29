"""End-to-end CLI tests for the ``survey`` command.

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual typer app.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from rich.console import Console

from lode.cli import app
from lode.cli.render import ACCESSIBLE_LIGHT_INTENT_COLORS, RenderOptions
from lode.ingestion.pipeline import DetectResult
from tests.conftest import fake_embedder, runner


def test_survey_reports_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    report = tmp_path / "a.txt"
    report.write_text("hello world")

    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])
    report.write_text("changed content!")

    survey = runner.invoke(app, ["--workspace", str(tmp_path), "survey"])
    assert survey.exit_code == 0, survey.output
    # Human-readable output is presentation, not a stable contract: assert only
    # that the changed file is reported, not the exact formatting.
    assert "a.txt" in survey.output


def test_survey_without_index_reports_all_new_and_creates_no_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Survey with no index classifies everything as new without creating a db.

    It must not touch the embedder (no endpoint needed) and must not create
    the index database — that is mine's job.
    """
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("second file")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["summary"]["new"] == 2
    assert payload["summary"]["pending"] == 2
    assert payload["summary"]["changed"] == 0
    # No index database was created by a read-only survey.
    assert not (tmp_path / ".lode" / "index.db").exists()


def test_survey_uses_configured_no_color(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured output.no_color flows into the render options."""
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text("[app.output]\nno_color = true\n")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    captured: RenderOptions | None = None

    def fake_render(
        workspace: Path,
        result: DetectResult,
        *,
        options: RenderOptions | None = None,
        console: Console | None = None,
    ) -> None:
        nonlocal captured
        captured = options

    monkeypatch.setattr("lode.cli.render_survey", fake_render)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey"])
    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured.no_color is True


def test_survey_palette_flag_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The --palette flag overrides the configured palette."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[app.output]\npalette = "ansi"\n')

    captured: RenderOptions | None = None

    def fake_render(
        workspace: Path,
        result: DetectResult,
        *,
        options: RenderOptions | None = None,
        console: Console | None = None,
    ) -> None:
        nonlocal captured
        captured = options

    monkeypatch.setattr("lode.cli.render_survey", fake_render)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey", "--palette", "accessible_light"])
    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured.intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS


def test_survey_no_color_flag_wins_over_palette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-color wins over --palette: colour off, palette still resolved."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello world")

    captured: RenderOptions | None = None

    def fake_render(
        workspace: Path,
        result: DetectResult,
        *,
        options: RenderOptions | None = None,
        console: Console | None = None,
    ) -> None:
        nonlocal captured
        captured = options

    monkeypatch.setattr("lode.cli.render_survey", fake_render)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey", "--no-color", "--palette", "accessible_light"])
    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured.no_color is True
    assert captured.intent_colors == ACCESSIBLE_LIGHT_INTENT_COLORS


def test_survey_json_reports_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    (tmp_path / "a.txt").write_text("changed content!")
    (tmp_path / "b.txt").write_text("brand new file")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["command"] == "survey"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    assert payload["workspace"] == str(tmp_path)
    assert payload["summary"] == {
        "unchanged": 0,
        "new": 1,
        "changed": 1,
        "missing": 0,
        "renamed": 0,
        "skipped": 0,
        "pending": 2,
    }
    assert payload["paths"]["new"] == ["b.txt"]
    assert payload["paths"]["changed"] == ["a.txt"]
    assert payload["paths"]["missing"] == []
    assert payload["paths"]["renamed"] == []
    assert payload["paths"]["unchanged"] == []


def test_survey_json_reports_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    (tmp_path / "a.txt").rename(tmp_path / "b.txt")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["summary"]["renamed"] == 1
    assert payload["summary"]["new"] == 0
    assert payload["summary"]["missing"] == 0
    assert payload["paths"]["renamed"] == [{"from": "a.txt", "to": "b.txt"}]
    assert payload["paths"]["new"] == []
    assert payload["paths"]["missing"] == []


def test_survey_json_error_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Corrupt the schema version so the store refuses to open.
    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", ("999",))
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "survey"
    assert payload["success"] is False
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "schema_version"
