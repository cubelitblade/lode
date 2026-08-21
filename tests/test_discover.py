"""Tests for file discovery and ignore rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from lode.ingestion.discover import discover


def test_discovers_files_as_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("b")

    assert discover(tmp_path) == [Path("a.txt"), Path("docs/b.md")]


def test_always_ignores_lode_data_directory(tmp_path: Path) -> None:
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "index.db").write_text("x")
    (tmp_path / "keep.txt").write_text("k")

    assert discover(tmp_path) == [Path("keep.txt")]


def test_applies_custom_glob_patterns(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("k")
    (tmp_path / "drop.tmp").write_text("t")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "also.tmp").write_text("t")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("c")

    result = discover(tmp_path, patterns=[".git/**", "*.tmp"])
    assert result == [Path("keep.txt")]


def test_bare_directory_pattern_ignores_subtree(tmp_path: Path) -> None:
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "x.txt").write_text("x")
    (tmp_path / "vendor" / "deep").mkdir()
    (tmp_path / "vendor" / "deep" / "y.txt").write_text("y")
    (tmp_path / "main.txt").write_text("m")

    assert discover(tmp_path, patterns=["vendor"]) == [Path("main.txt")]


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")
