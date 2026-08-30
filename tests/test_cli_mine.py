"""End-to-end CLI tests for the ``mine`` command.

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual typer app.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lode.cli import app
from lode.relpath import to_native, to_rel
from tests.conftest import (
    dimension_mismatch_embedder,
    failing_embedder,
    fake_embedder,
    other_model_embedder,
    runner,
)


def test_mine_then_prospect_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(
        "The experiment showed strong quantum entanglement in the third group."
    )

    mine = runner.invoke(app, ["--workspace", str(tmp_path), "mine"])
    assert mine.exit_code == 0, mine.output
    assert "+ added 1" in mine.output

    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "entanglement"])
    assert prospect.exit_code == 0, prospect.output
    # The render layer shows OS-native paths.
    assert str(to_native(to_rel("docs/report.txt"))) in prospect.output
    assert "quantum entanglement" in prospect.output
    # The short chunk id (blake3: prefix stripped) is appended to the line.
    assert "#" in prospect.output
    assert "blake3:" not in prospect.output


def test_mine_from_scratch_flag_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    mine = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--from-scratch"])
    assert mine.exit_code == 0, mine.output
    assert "+ added 1" in mine.output


def test_mine_with_no_indexable_files_creates_no_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mine with only unsupported files reports Nothing to do. without a db.

    It must not create the index database or touch the embedder when there is
    nothing to embed.
    """
    (tmp_path / "pic.png").write_bytes(b"nope")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    assert result.exit_code == 0, result.output
    assert "Nothing to do." in result.output
    assert not (tmp_path / ".lode" / "index.db").exists()


def test_mine_uses_chunking_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long file is split into more chunks with a small chunk_size."""
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    # Config is loaded relative to the CWD (see lode.toml.example), so run
    # the CLI from the workspace root.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text("[chunking]\nsize = 20\noverlap = 5\n")
    (tmp_path / "a.txt").write_text("word " * 100)

    mine = runner.invoke(app, ["--workspace", str(tmp_path), "mine"])
    assert mine.exit_code == 0, mine.output
    assert "+ added 1" in mine.output

    # With chunk_size=20 the file must have been split into multiple chunks.
    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    assert n_chunks > 1


def test_mine_json_reports_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    (tmp_path / "a.txt").rename(tmp_path / "b.txt")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["success"] is True
    assert payload["summary"]["renamed"] == 1
    assert payload["summary"]["added"] == 0
    assert payload["summary"]["removed"] == 0
    assert payload["paths"]["renamed"] == [{"from": "a.txt", "to": "b.txt"}]
    assert payload["paths"]["added"] == []
    assert payload["paths"]["removed"] == []


def test_mine_json_reports_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["command"] == "mine"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    assert payload["from_scratch"] is False
    assert payload["workspace"] == tmp_path.as_posix()
    assert payload["summary"] == {
        "added": 2,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
        "renamed": 0,
        "skipped": 0,
    }
    assert set(payload["paths"]["added"]) == {"a.txt", "b.txt"}
    assert payload["paths"]["updated"] == []
    assert payload["paths"]["removed"] == []
    assert payload["paths"]["renamed"] == []
    assert payload["failed"] == []


def test_mine_from_scratch_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--from-scratch", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["from_scratch"] is True
    assert payload["summary"]["added"] == 1


def test_mine_json_reports_failed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", failing_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["failed"] == [{"path": "a.txt", "error": "embedding endpoint is down"}]
    assert payload["summary"]["added"] == 0


def test_mine_json_model_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    monkeypatch.setattr("lode.cli.build_embedder", other_model_embedder)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "mine"
    assert payload["success"] is False
    assert payload["error"]["code"] == "model_mismatch"


def test_mine_json_schema_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", ("999",))
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "schema_version"


def test_mine_from_scratch_schema_version_exempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--from-scratch exempts schema_version mismatch: old db is reset, then re-mined."""
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Corrupt schema version so a plain open would refuse.
    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", ("999",))
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--from-scratch", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["from_scratch"] is True
    assert payload["summary"]["added"] == 1


def test_mine_from_scratch_tokenizer_mismatch_exempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--from-scratch exempts tokenizer mismatch: old db is reset, then re-mined."""
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    # Build an index with trigram tokenizer.
    (tmp_path / "lode.toml").write_text('[fts]\nstrategy = "trigram"\n')
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Switch back to default (simple) without --from-scratch → blocked.
    (tmp_path / "lode.toml").write_text('[fts]\nstrategy = "simple"\n')
    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine"])
    assert result.exit_code != 0

    # Switch with --from-scratch → reset and re-mine.
    result = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--from-scratch", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["from_scratch"] is True
    assert payload["summary"]["added"] == 1


def test_mine_from_scratch_dimension_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # A dimension change is allowed with an explicit --from-scratch.
    monkeypatch.setattr("lode.cli.build_embedder", dimension_mismatch_embedder)
    mine = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--from-scratch"])
    assert mine.exit_code == 0, mine.output

    # After re-mine the index is 99-dim; prospect with the same embedder works.
    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello"])
    assert prospect.exit_code == 0, prospect.output
