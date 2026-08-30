"""Tests for file discovery and ignore rules."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from lode.ingestion.discover import discover


def test_discovers_files_as_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("b")

    assert discover(tmp_path) == [PurePosixPath("a.txt"), PurePosixPath("docs/b.md")]


def test_always_ignores_lode_data_directory(tmp_path: Path) -> None:
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "index.db").write_text("x")
    (tmp_path / "keep.txt").write_text("k")

    assert discover(tmp_path) == [PurePosixPath("keep.txt")]


def test_lodeignore_is_first_class(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("k")
    (tmp_path / "drop.log").write_text("l")
    (tmp_path / ".lodeignore").write_text("*.log\n")

    assert discover(tmp_path) == [PurePosixPath("keep.txt")]


def test_loads_configured_ignore_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("k")
    (tmp_path / "drop.tmp").write_text("t")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "also.tmp").write_text("t")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("c")
    (tmp_path / ".gitignore").write_text(".git/**\n*.tmp\n")

    result = discover(tmp_path, ignore_files=[".gitignore"])

    assert result == [PurePosixPath("keep.txt")]


def test_bare_directory_pattern_ignores_subtree(tmp_path: Path) -> None:
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "x.txt").write_text("x")
    (tmp_path / "vendor" / "deep").mkdir()
    (tmp_path / "vendor" / "deep" / "y.txt").write_text("y")
    (tmp_path / "main.txt").write_text("m")
    (tmp_path / ".lodeignore").write_text("vendor\n")

    assert discover(tmp_path) == [PurePosixPath("main.txt")]


def test_gitignore_negation(tmp_path: Path) -> None:
    (tmp_path / "drop.log").write_text("l")
    (tmp_path / "keep.log").write_text("k")
    (tmp_path / ".lodeignore").write_text("*.log\n!keep.log\n")

    assert discover(tmp_path) == [PurePosixPath("keep.log")]


def test_ignore_files_themselves_are_excluded(tmp_path: Path) -> None:
    (tmp_path / ".lodeignore").write_text("*.tmp\n")
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "a.txt").write_text("a")

    result = discover(tmp_path, ignore_files=[".gitignore"])

    assert result == [PurePosixPath("a.txt")]


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")
