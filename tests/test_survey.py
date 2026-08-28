"""Tests for the ingestion survey pipeline (detect/classify).

Hermetic: real temp files + FakeEmbedder, no network.
"""

from __future__ import annotations

from pathlib import Path

from lode.index import FileStatus, Store
from lode.ingestion.pipeline import classify, detect_changes
from tests.conftest import run_sync, write
from tests.fakes import FakeEmbedder


def test_survey_detects_changes_without_embedding(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    path.write_text("changed!")

    summary = detect_changes(store, tmp_path)

    assert summary.changed == 1
    assert summary.pending == 1
    # The embedder was never needed, so even a broken one is fine.
    detect_changes(store, tmp_path)


def test_survey_reports_new_missing_and_skipped(store: Store, tmp_path: Path) -> None:
    indexed = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    indexed.unlink()
    write(tmp_path, "b.txt", "brand new file")
    write(tmp_path, "pic.png", "nope")

    summary = detect_changes(store, tmp_path)

    assert summary.new == 1
    assert summary.missing == 1
    assert summary.skipped == 1
    assert summary.pending == 2  # new + missing


def test_survey_reports_residual_stale_marker_as_changed(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    # Simulate a failed sync that left a stale marker without any disk change.
    store.mark_stale("a.txt")

    summary = detect_changes(store, tmp_path)

    # Stat matches but the file is still stale: it must be retried, so it is
    # folded into changed (not unchanged) — this is the C2 fix.
    assert summary.unchanged == 0
    assert summary.changed == 1
    assert summary.pending == 1
    assert summary.dirty is True


def test_detect_marks_changed_stale_and_missing_not(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    write(tmp_path, "b.txt", "second file content")
    run_sync(store, tmp_path, FakeEmbedder())
    path.write_text("changed content")
    (tmp_path / "b.txt").unlink()

    summary = detect_changes(store, tmp_path)

    # Changed file is marked stale; missing file is not (no snapshot to flag).
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.STALE
    assert summary.changed == 1
    assert summary.missing == 1
    assert summary.dirty is True


def test_detect_residual_stale_not_remark(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    store.mark_stale("a.txt")

    detect_changes(store, tmp_path)

    # Residual stale is folded into changed but not re-marked (already stale).
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.STALE


def test_classify_empty_snapshot_is_all_new_and_side_effect_free(tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    write(tmp_path, "b.txt", "second file content")
    write(tmp_path, "pic.png", "nope")

    summary = classify({}, tmp_path)

    # No index yet: every supported file is new, nothing is stale, no missing.
    assert summary.new == 2
    assert summary.changed == 0
    assert summary.unchanged == 0
    assert summary.missing == 0
    assert summary.skipped == 1
    assert summary.stale_paths == []
    assert summary.pending == 2
    assert summary.dirty is False


def test_classify_reports_stale_paths_without_touching_store(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    path.write_text("changed content")

    indexed = {file.path: file for file in store.list_files()}
    summary = classify(indexed, tmp_path)

    # The changed path is reported as a side effect to apply, but classify
    # itself must not have flipped the marker (that is detect_changes' job).
    assert summary.stale_paths == ["a.txt"]
    assert summary.changed == 1
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.FRESH
