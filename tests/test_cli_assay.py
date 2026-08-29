"""End-to-end CLI tests for the ``assay`` command (and its ``analyze`` alias).

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual typer app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lode.cli import app
from lode.ingestion import chunk_digest
from tests.conftest import fake_embedder, runner


def test_assay_explains_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    digest = chunk_digest(text)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "why", digest, "entanglement"])
    assert result.exit_code == 0, result.output
    assert "Query: entanglement" in result.output
    assert "semantic" in result.output
    assert "lexical" in result.output
    assert "Score:" in result.output
    # Only the short stub is shown, never the full blake3 prefix.
    assert "blake3:" not in result.output


def test_assay_accepts_short_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    short = chunk_digest(text).removeprefix("blake3:")[:12]
    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "why", short, "entanglement"])
    assert result.exit_code == 0, result.output
    assert "Query: entanglement" in result.output
    assert "Score:" in result.output


def test_assay_analyze_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "analyze", "why", chunk_digest(text), "entanglement"])
    assert result.exit_code == 0, result.output
    assert "Query: entanglement" in result.output
    assert "Score:" in result.output


def test_assay_missing_digest_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    absent = chunk_digest("this text is not indexed")
    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "why", absent, "hello"])
    assert result.exit_code != 0


def test_assay_invalid_digest_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "why", "not-a-digest!", "hello"])
    assert result.exit_code != 0


def test_assay_without_index_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "why", "deadbeef", "hello"])
    assert result.exit_code != 0


def test_assay_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(
        app, ["--workspace", str(tmp_path), "assay", "why", chunk_digest(text), "entanglement", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "assay"
    assert payload["success"] is True
    explanation = payload["explanation"]
    assert set(explanation) == {
        "digest",
        "paths",
        "heading",
        "page",
        "seq",
        "sources",
        "norm",
        "fusion",
        "combined",
        "rank",
        "in_results",
        "top_k",
    }
    assert set(explanation["sources"]) == {"semantic", "lexical"}
    for source in explanation["sources"].values():
        assert set(source) == {"status", "pool_size", "raw", "prepared", "pool_rank"}
    assert explanation["sources"]["semantic"]["status"] == "matched"
    assert explanation["sources"]["lexical"]["status"] == "matched"
    assert explanation["norm"] == {"name": "softmax", "params": {"temperature": 1.0}}
    assert explanation["fusion"] == {
        "name": "linear",
        "params": {"weights": {"semantic": 0.7, "lexical": 0.3}},
    }
    assert explanation["sources"]["semantic"]["raw"] is not None
    assert explanation["sources"]["lexical"]["raw"] is not None
    assert explanation["in_results"] is True
    assert explanation["rank"] == 1


def test_assay_json_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(
        app, ["--workspace", str(tmp_path), "assay", "why", chunk_digest("not indexed"), "hello", "--json"]
    )
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "not_found"


def test_assay_how_shows_tokenization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    digest = chunk_digest(text)
    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "how", digest])
    assert result.exit_code == 0, result.output
    # Tokenizer metadata and the term stream are shown.
    assert "simple" in result.output
    assert "quantum" in result.output
    assert "Terms" in result.output
    # Provenance is shown; the full content address is not.
    assert "a.txt" in result.output
    assert "blake3:" not in result.output


def test_assay_how_analyze_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "analyze", "how", chunk_digest(text)])
    assert result.exit_code == 0, result.output
    assert "Terms" in result.output


def test_assay_how_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "how", chunk_digest(text), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "assay"
    assert payload["success"] is True
    assert payload["tokenizer"] == {"strategy": "simple", "tokenize_clause": "simple"}
    assert payload["text"] == text
    assert payload["char_count"] == len(text)
    assert payload["token_count"] == len(payload["tokens"])
    assert payload["term_count"] == len(payload["terms"])
    # The raw token stream reflects index-side normalization (case folding).
    assert "quantum" in payload["tokens"]
    assert "QUANTUM" not in payload["tokens"]
    # The structured terms carry surfaces with their variants.
    assert all(set(term) == {"surface", "variants"} for term in payload["terms"])
    assert {term["surface"] for term in payload["terms"]} >= {"quantum", "entanglement"}
    assert payload["paths"] == [{"path": "a.txt", "state": "fresh"}]


def test_assay_how_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["--workspace", str(tmp_path), "mine"])

    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "how", chunk_digest("not indexed"), "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "not_found"


def test_assay_how_without_index_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", fake_embedder)
    (tmp_path / "a.txt").write_text("hello")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "assay", "how", "deadbeef", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "no_index"
