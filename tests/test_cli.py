"""End-to-end CLI tests: mine -> prospect round trip.

The real embedder (network) is replaced with a FakeEmbedder via
monkeypatch; everything else runs through the actual typer app.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lode.cli import app
from lode.config import EmbeddingConfig
from tests.fakes import FakeEmbedder

runner = CliRunner()


def _fake_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return FakeEmbedder()


def _other_model_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return FakeEmbedder(model_id="other-model")


def test_mine_then_prospect_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(
        "The experiment showed strong quantum entanglement in the third group."
    )

    mine = runner.invoke(app, ["mine", str(tmp_path)])
    assert mine.exit_code == 0, mine.output
    assert "1 added" in mine.output

    prospect = runner.invoke(app, ["prospect", "entanglement", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output
    assert "docs/report.txt" in prospect.output
    assert "quantum entanglement" in prospect.output


def test_survey_reports_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    report = tmp_path / "a.txt"
    report.write_text("hello world")

    runner.invoke(app, ["mine", str(tmp_path)])
    report.write_text("changed content!")

    survey = runner.invoke(app, ["survey", str(tmp_path)])
    assert survey.exit_code == 0, survey.output
    assert "1 changed" in survey.output
    assert "pending" in survey.output


def test_mine_rebuild_flag_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    mine = runner.invoke(app, ["mine", "--rebuild", str(tmp_path)])
    assert mine.exit_code == 0, mine.output
    assert "1 added" in mine.output


def test_model_mismatch_blocks_prospect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Same store, different model -> refused until --rebuild.
    monkeypatch.setattr("lode.cli.build_embedder", _other_model_embedder)
    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code != 0
    assert "different model" in prospect.output

    mine = runner.invoke(app, ["mine", "--rebuild", str(tmp_path)])
    assert mine.exit_code == 0, mine.output

    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output


def test_search_alias_is_hidden_and_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["search", "hello", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "a.txt" in result.output

    help_out = runner.invoke(app, ["--help"]).output
    # Hidden aliases (status/index/search) must not show in the command list.
    assert "│ search " not in help_out
    assert "│ status " not in help_out
    assert "│ index " not in help_out
    assert "│ survey " in help_out
    assert "│ mine " in help_out
    assert "│ prospect " in help_out
