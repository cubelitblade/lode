"""Tests for the ingestion pipeline: survey (detect) and sync (update).

Hermetic: real temp files + FakeEmbedder, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lode.index.store import FileStatus, Store
from lode.ingestion.pipeline import survey_workspace, sync
from lode.ingestion.split import RecursiveTextSplitter
from tests.fakes import FailingEmbedder, FakeEmbedder


@pytest.fixture
def store(tmp_path: Path) -> Store:
    # Keep the db under .lode/ like the real CLI, so the WAL sidecar files
    # are ignored by discover and don't pollute the counts.
    return Store(tmp_path / ".lode" / "index.db", FakeEmbedder())


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


SPLITTER = RecursiveTextSplitter(chunk_size=50, chunk_overlap=5)


def test_sync_indexes_new_files(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")

    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    assert result.added == 1
    files = store.list_files()
    assert len(files) == 1
    assert files[0].path == "a.txt"
    assert files[0].status is FileStatus.CURRENT


def test_sync_is_idempotent(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    assert result.added == 0
    assert result.updated == 0
    assert result.unchanged == 1


def test_sync_reindexes_changed_file(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)
    path.write_text("totally different text now")

    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    assert result.updated == 1
    assert store.get_file("a.txt") is not None


def test_sync_removes_files_gone_from_disk(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)
    path.unlink()

    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    assert result.removed == 1
    assert store.list_files() == []


def test_sync_skips_unsupported_formats(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "image.png", "not text at all")
    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER)
    assert result.skipped == 1
    assert store.list_files() == []


def test_sync_failure_marks_file_stale_and_keeps_going(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    write(tmp_path, "b.txt", "second file content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    # Change both, then sync with a broken embedder: both fail, both stale.
    (tmp_path / "a.txt").write_text("changed a")
    (tmp_path / "b.txt").write_text("changed b")
    result = sync(store, tmp_path, FailingEmbedder(), SPLITTER)

    assert len(result.failed) == 2
    file_a = store.get_file("a.txt")
    file_b = store.get_file("b.txt")
    assert file_a is not None
    assert file_b is not None
    assert file_a.status is FileStatus.STALE
    assert file_b.status is FileStatus.STALE


def test_sync_retries_stale_files_even_if_unchanged(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    path.write_text("changed content here")
    sync(store, tmp_path, FailingEmbedder(), SPLITTER)
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.STALE

    # Same mtime+size as the failed run, but the retry must still re-embed.
    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER)

    assert result.updated == 1
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.CURRENT


def test_survey_detects_changes_without_embedding(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)
    path.write_text("changed!")

    summary = survey_workspace(store, tmp_path)

    assert summary.changed == 1
    assert summary.pending == 1
    # The embedder was never needed, so even a broken one is fine.
    survey_workspace(store, tmp_path)


def test_survey_reports_new_missing_and_skipped(store: Store, tmp_path: Path) -> None:
    indexed = write(tmp_path, "a.txt", "hello world content")
    sync(store, tmp_path, FakeEmbedder(), SPLITTER)
    indexed.unlink()
    write(tmp_path, "b.txt", "brand new file")
    write(tmp_path, "pic.png", "nope")

    summary = survey_workspace(store, tmp_path)

    assert summary.new == 1
    assert summary.missing == 1
    assert summary.skipped == 1
    assert summary.pending == 2  # new + missing
