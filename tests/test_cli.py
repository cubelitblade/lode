"""End-to-end CLI tests: mine -> prospect round trip.

The real embedder (network) is replaced with a FakeEmbedder via
monkeypatch; everything else runs through the actual typer app.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lode.cli import app
from lode.config import EmbeddingConfig
from lode.ingestion import chunk_id
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
    # The short chunk id (blake3: prefix stripped) is appended to the line.
    assert "#" in prospect.output
    assert "blake3:" not in prospect.output


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


def test_mine_uses_chunking_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long file is split into more chunks with a small chunk_size."""
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    # Config is loaded relative to the CWD (see lode.toml.example), so run
    # the CLI from the workspace root.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text("[chunking]\nsize = 20\noverlap = 5\n")
    (tmp_path / "a.txt").write_text("word " * 100)

    mine = runner.invoke(app, ["mine", str(tmp_path)])
    assert mine.exit_code == 0, mine.output
    assert "1 added" in mine.output

    # With chunk_size=20 the file must have been split into multiple chunks.
    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    assert n_chunks > 1


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


def test_prospect_warns_stale_files_outside_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Change a.txt so survey marks it stale; b.txt stays current.
    (tmp_path / "a.txt").write_text("hello world changed")
    runner.invoke(app, ["survey", str(tmp_path)])

    # top-k 1 keeps only the current hit (b.txt), so the stale file is not
    # in the result set.
    prospect = runner.invoke(app, ["prospect", "entanglement", str(tmp_path), "--top-k", "1"])
    assert prospect.exit_code == 0, prospect.output
    assert "stale files outside these results" in prospect.output
    assert "Run `lode mine`" in prospect.output


def test_prospect_warns_stale_files_in_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")
    runner.invoke(app, ["mine", str(tmp_path)])

    (tmp_path / "a.txt").write_text("hello world changed")
    runner.invoke(app, ["survey", str(tmp_path)])

    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output
    assert "results include stale files" in prospect.output
    assert "verify them before relying on them" in prospect.output
    assert "Run `lode mine`" in prospect.output


def test_dig_returns_full_chunk_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    digest = chunk_id(text)
    dig = runner.invoke(app, ["dig", digest, str(tmp_path)])
    assert dig.exit_code == 0, dig.output
    assert "docs/report.txt" in dig.output
    assert text in dig.output
    # Only the short stub is shown, never the full blake3 prefix.
    assert "blake3:" not in dig.output


def test_dig_accepts_short_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    short = chunk_id(text).removeprefix("blake3:")[:12]
    dig = runner.invoke(app, ["dig", short, str(tmp_path)])
    assert dig.exit_code == 0, dig.output
    assert text in dig.output
    assert "a.txt" in dig.output


def test_dig_accepts_bare_hex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    hex_digest = chunk_id(text).removeprefix("blake3:")
    dig = runner.invoke(app, ["dig", hex_digest, str(tmp_path)])
    assert dig.exit_code == 0, dig.output
    assert text in dig.output


def test_dig_missing_digest_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    absent = chunk_id("this text is not indexed")
    dig = runner.invoke(app, ["dig", absent, str(tmp_path)])
    assert dig.exit_code != 0
    assert "Dry hole" in dig.output


def test_dig_without_index_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")  # never mined

    dig = runner.invoke(app, ["dig", "deadbeef", str(tmp_path)])
    assert dig.exit_code != 0
    assert "Dry hole" in dig.output
    assert "run `lode mine`" in dig.output


def test_dig_invalid_digest_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    dig = runner.invoke(app, ["dig", "not-a-digest!", str(tmp_path)])
    assert dig.exit_code != 0
    assert "not a valid digest" in dig.output


def test_get_alias_is_hidden_and_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["get", chunk_id(text), str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert text in result.output

    help_out = runner.invoke(app, ["--help"]).output
    assert "│ get " not in help_out
    assert "│ dig " in help_out
