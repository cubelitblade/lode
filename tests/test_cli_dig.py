"""End-to-end CLI tests for the ``dig`` command (and its ``get`` alias).

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual typer app.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lode.cli import app
from lode.ingestion import chunk_digest
from lode.relpath import to_native, to_rel
from tests.conftest import fake_embedder, runner


def _mine_report_with_many_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Mine a single .txt file into several chunks and return one chunk id."""
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    monkeypatch.chdir(tmp_path)
    # Small chunks so one file yields several chunks in a single section.
    (tmp_path / "lode.toml").write_text("[chunking]\nsize = 50\noverlap = 10\n")
    (tmp_path / "report.txt").write_text(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi " * 12
    )
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    row = conn.execute("SELECT digest FROM chunks ORDER BY seq LIMIT 1").fetchone()
    conn.close()
    return row[0]


def test_dig_returns_full_chunk_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    digest = chunk_digest(text)
    dig = runner.invoke(app, ["--workspace", str(tmp_path), "dig", digest])
    assert dig.exit_code == 0, dig.output
    # The render layer shows OS-native paths.
    assert str(to_native(to_rel("docs/report.txt"))) in dig.output
    assert text in dig.output
    # Only the short stub is shown, never the full blake3 prefix.
    assert "blake3:" not in dig.output


def test_dig_accepts_short_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    short = chunk_digest(text).removeprefix("blake3:")[:12]
    dig = runner.invoke(app, ["--workspace", str(tmp_path), "dig", short])
    assert dig.exit_code == 0, dig.output
    assert text in dig.output
    assert "a.txt" in dig.output


def test_dig_accepts_bare_hex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    hex_digest = chunk_digest(text).removeprefix("blake3:")
    dig = runner.invoke(app, ["--workspace", str(tmp_path), "dig", hex_digest])
    assert dig.exit_code == 0, dig.output
    assert text in dig.output


def test_dig_missing_digest_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    absent = chunk_digest("this text is not indexed")
    dig = runner.invoke(app, ["--workspace", str(tmp_path), "dig", absent])
    assert dig.exit_code != 0


def test_dig_without_index_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")  # never mined

    dig = runner.invoke(app, ["--workspace", str(tmp_path), "dig", "deadbeef"])
    assert dig.exit_code != 0


def test_dig_invalid_digest_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    dig = runner.invoke(app, ["--workspace", str(tmp_path), "dig", "not-a-digest!"])
    assert dig.exit_code != 0


def test_dig_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "a.txt").write_text(text)
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", chunk_digest(text), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "dig"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    window = payload["window"]
    assert window["center_seq"] == 0
    assert window["radius"] == 0
    assert len(window["chunks"]) == 1
    chunk = window["chunks"][0]
    assert chunk["digest"] == chunk_digest(text)
    assert chunk["paths"] == [{"path": "a.txt", "state": "fresh"}]
    assert chunk["heading"] == ""
    assert chunk["page"] is None
    assert chunk["text"] == text


def test_dig_json_accepts_short_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    short = chunk_digest(text).removeprefix("blake3:")[:12]
    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", short, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    chunk = payload["window"]["chunks"][0]
    assert chunk["digest"] == chunk_digest(text)
    assert chunk["text"] == text


def test_dig_json_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(
        app, ["--workspace", str(tmp_path), "dig", chunk_digest("this text is not indexed"), "--json"]
    )
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "not_found"


def test_dig_json_invalid_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("quantum")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", "not-a-digest!", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "invalid_digest"


def test_dig_json_no_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", "deadbeef", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "no_index"


def test_dig_radius_returns_section_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _mine_report_with_many_chunks(tmp_path, monkeypatch)

    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", digest, "--radius", "1"])
    assert result.exit_code == 0, result.output
    short = digest.removeprefix("blake3:")[:12]
    assert f"Dug {short} with radius 1" in result.output
    # The center chunk is marked and the neighbour card carries its own text.
    assert "0 · center" in result.output
    assert "iota kappa lambda" in result.output


def test_dig_json_radius_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _mine_report_with_many_chunks(tmp_path, monkeypatch)

    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", digest, "--radius", "1", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "dig"
    assert payload["success"] is True
    window = payload["window"]
    assert window["center_seq"] == 0
    assert window["radius"] == 1
    assert isinstance(window["chunks"], list)
    assert len(window["chunks"]) >= 1
    center = window["chunks"][0]
    assert center["seq"] == 0
    assert center["digest"].startswith("blake3:")
    assert "text" in center
    assert "heading" in center
    assert "paths" in center
    # The window holds at least one distinct neighbor chunk beyond the center.
    assert any(chunk["digest"] != digest for chunk in window["chunks"])


def test_dig_json_default_single_chunk_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "dig", chunk_digest(text), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    window = payload["window"]
    assert window["center_seq"] == 0
    assert window["radius"] == 0
    assert len(window["chunks"]) == 1


def test_get_alias_is_hidden_and_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "get", chunk_digest(text)])
    assert result.exit_code == 0, result.output
    assert text in result.output

    # Hidden aliases must be registered but not listed as visible commands.
    # Checked via the Typer registry, not help rendering, so the assertion
    # does not depend on presentation (colours, borders, wrapping).
    visible = {c.name for c in app.registered_commands if not c.hidden}
    hidden = {c.name for c in app.registered_commands if c.hidden}
    assert "get" in hidden
    assert "dig" in visible
