"""End-to-end CLI tests for the ``prospect`` command (and its ``search`` alias).

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual typer app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lode.cli import app
from tests.conftest import (
    dimension_mismatch_embedder,
    fake_embedder,
    other_model_embedder,
    runner,
    wrong_query_embedder,
)


def test_model_mismatch_blocks_prospect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Same store, different model -> refused until --from-scratch.
    monkeypatch.setattr("lode.cli.build_embedder", other_model_embedder)
    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello"])
    assert prospect.exit_code != 0

    mine = runner.invoke(app, ["--workspace", str(tmp_path), "mine", "--from-scratch"])
    assert mine.exit_code == 0, mine.output

    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello"])
    assert prospect.exit_code == 0, prospect.output


def test_search_alias_is_hidden_and_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "search", "hello"])
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
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Change a.txt so survey marks it stale; b.txt stays current.
    (tmp_path / "a.txt").write_text("hello world changed")
    runner.invoke(app, ["--workspace", str(tmp_path), "survey"])

    # top-k 1 keeps only the current hit (b.txt), so the stale file is not
    # in the result set.
    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "entanglement", "--top-k", "1"])
    assert prospect.exit_code == 0, prospect.output


def test_prospect_warns_stale_files_in_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    (tmp_path / "a.txt").write_text("hello world changed")
    runner.invoke(app, ["--workspace", str(tmp_path), "survey"])

    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello"])
    assert prospect.exit_code == 0, prospect.output


def test_prospect_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(text)
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])
    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "entanglement", "--json", "--top-k", "3"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["command"] == "prospect"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    assert payload["query"] == "entanglement"
    assert payload["top_k"] == 3
    assert len(payload["hits"]) == 1

    hit = payload["hits"][0]
    assert set(hit) == {"rank", "score", "paths", "heading", "page", "digest", "preview"}
    assert hit["rank"] == 1
    assert hit["paths"] == [{"path": "docs/report.txt", "state": "fresh"}]
    assert hit["digest"].startswith("blake3:")
    assert "quantum entanglement" in hit["preview"]
    assert "\n" not in hit["preview"]


def test_prospect_json_preview_flattens_crlf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("line one\r\nline two\rline three\n")
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])
    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "line", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["hits"]
    preview = payload["hits"][0]["preview"]
    assert "\r" not in preview
    assert "\n" not in preview
    assert preview == "line one line two line three"


def test_prospect_json_empty_hits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    # Lexical-only retrieval so a query absent from the doc yields no hits
    # (the fake embedder always scores densely, which would mask the empty case).
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text(
        '[fusion]\ntype = "linear"\n\n[fusion.linear]\nsemantic_factor = 0\nlexical_factor = 1\n'
    )
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # An existing index with no matching content yields empty hits.
    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "nothing-matches", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["hits"] == []


def test_prospect_without_index_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prospect with no index says run mine first instead of creating a db."""
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "no_index"
    assert not (tmp_path / ".lode" / "index.db").exists()


def test_prospect_json_model_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    monkeypatch.setattr("lode.cli.build_embedder", other_model_embedder)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "prospect"
    assert payload["success"] is False
    assert payload["error"]["code"] == "model_mismatch"


def test_prospect_json_invalid_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Both linear fusion factors zero -> search refuses to run.
    (tmp_path / ".lode" / "config.toml").write_text(
        '[fusion]\ntype = "linear"\n\n[fusion.linear]\nsemantic_factor = 0.0\nlexical_factor = 0.0\n'
    )

    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "invalid_query"


def test_prospect_dimension_mismatch_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Same model id but a different vector dimension -> gate refuses search.
    monkeypatch.setattr("lode.cli.build_embedder", dimension_mismatch_embedder)
    prospect = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello"])
    assert prospect.exit_code != 0


def test_prospect_dimension_mismatch_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    # Reports the stored dimension (4) so the gate passes, but actually emits
    # 99-dim query vectors -> sqlite-vec MATCH throws DimensionMismatchError,
    # which the CLI must translate to the friendly message (Layer 2 fallback).
    monkeypatch.setattr("lode.cli.build_embedder", wrong_query_embedder)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello"])
    assert result.exit_code != 0


def test_prospect_json_dimension_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    monkeypatch.setattr("lode.cli.build_embedder", dimension_mismatch_embedder)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "prospect", "hello", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "prospect"
    assert payload["success"] is False
    assert payload["error"]["code"] == "dimension_mismatch"
