"""Tests for the ingestion pipeline: survey (detect) and sync (update).

Hermetic: real temp files + FakeEmbedder, no network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lode.index.search import search
from lode.index.store import FileStatus, Store
from lode.ingestion import Chunk, Segment
from lode.ingestion.pipeline import FailedFile, SyncSummary, classify, detect_changes, sync
from lode.ingestion.split import RecursiveSegmentSplitter, SegmentSplitter
from tests.fakes import FailingEmbedder, FakeEmbedder, make_docx_bytes


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


SPLITTER = RecursiveSegmentSplitter(chunk_size=50, chunk_overlap=5)


def run_sync(
    store: Store,
    tmp_path: Path,
    embedder: FakeEmbedder | FailingEmbedder,
    *,
    report: Callable[[int, int, str], None] | None = None,
) -> SyncSummary:
    """detect then sync — the two-stage shape the CLI now uses."""
    detect = detect_changes(store, tmp_path)
    return sync(store, tmp_path, embedder, SPLITTER, detect=detect, report=report)


def test_sync_indexes_new_files(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")

    result = run_sync(store, tmp_path, FakeEmbedder())

    assert result.added == 1
    files = store.list_files()
    assert len(files) == 1
    assert files[0].path == "a.txt"
    assert files[0].status is FileStatus.FRESH


def test_sync_is_idempotent(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())

    result = run_sync(store, tmp_path, FakeEmbedder())

    assert result.added == 0
    assert result.updated == 0
    assert result.unchanged == 1


def test_sync_reports_progress(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    write(tmp_path, "b.txt", "more content here")

    calls: list[tuple[int, int, str]] = []
    result = run_sync(
        store,
        tmp_path,
        FakeEmbedder(),
        report=lambda done, total, path: calls.append((done, total, path)),
    )

    assert result.added == 2
    # One report before each file to embed (0-based count), then a completion report.
    assert [c[0] for c in calls] == [0, 1, 2]
    assert calls[0][1] == 2 and calls[1][1] == 2
    assert {c[2] for c in calls[:2]} == {"a.txt", "b.txt"}


def test_sync_reindexes_changed_file(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    path.write_text("totally different text now")

    result = run_sync(store, tmp_path, FakeEmbedder())

    assert result.updated == 1
    assert store.get_file("a.txt") is not None


def test_sync_removes_files_gone_from_disk(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())
    path.unlink()

    result = run_sync(store, tmp_path, FakeEmbedder())
    assert result.removed == 1
    assert store.list_files() == []


def test_sync_skips_unsupported_formats(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "image.png", "not text at all")
    result = run_sync(store, tmp_path, FakeEmbedder())
    assert result.skipped == 1
    assert store.list_files() == []


def test_sync_failure_marks_file_stale_and_keeps_going(store: Store, tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "hello world content")
    write(tmp_path, "b.txt", "second file content")
    run_sync(store, tmp_path, FakeEmbedder())

    # Change both, then sync with a broken embedder: both fail, both stale.
    (tmp_path / "a.txt").write_text("changed a")
    (tmp_path / "b.txt").write_text("changed b")
    result = run_sync(store, tmp_path, FailingEmbedder())

    assert len(result.failed) == 2
    assert all(isinstance(failure, FailedFile) for failure in result.failed)
    assert {failure.path for failure in result.failed} == {"a.txt", "b.txt"}
    assert all(failure.error for failure in result.failed)
    file_a = store.get_file("a.txt")
    file_b = store.get_file("b.txt")
    assert file_a is not None
    assert file_b is not None
    assert file_a.status is FileStatus.STALE
    assert file_b.status is FileStatus.STALE


def test_sync_reraises_unexpected_error(store: Store, tmp_path: Path) -> None:
    """A programming error in the splitter must propagate, not be swallowed.

    F2: the per-file tolerance only covers *domain* errors (extraction,
    embedding, store). A bug in the splitter is a coding error — silently
    downgrading it to a stale marker would hide the regression.
    """
    write(tmp_path, "a.txt", "hello world content")

    class _BrokenSplitter(SegmentSplitter):
        def split_segments(self, segments: list[Segment]) -> list[Chunk]:
            raise RuntimeError("splitter bug")

    detect = detect_changes(store, tmp_path)
    with pytest.raises(RuntimeError, match="splitter bug"):
        sync(store, tmp_path, FakeEmbedder(), _BrokenSplitter(), detect=detect)


def test_sync_corrupt_docx_fails_and_keeps_going(store: Store, tmp_path: Path) -> None:
    """A malformed docx is a domain error: the file fails, the run continues,
    and it is not a programming error. A brand-new file that fails on first
    index leaves no row behind (mark_stale on an unknown path is a no-op)."""
    (tmp_path / "bad.docx").write_bytes(b"this is not a real docx")

    result = run_sync(store, tmp_path, FakeEmbedder())

    assert len(result.failed) == 1
    assert result.failed[0].path == "bad.docx"
    assert result.failed[0].error
    # Never successfully indexed, so no snapshot row exists to flag stale.
    assert store.get_file("bad.docx") is None


def test_sync_retries_stale_files_even_if_unchanged(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())

    path.write_text("changed content here")
    run_sync(store, tmp_path, FailingEmbedder())
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.STALE

    # Same mtime+size as the failed run, but the retry must still re-embed.
    result = run_sync(store, tmp_path, FakeEmbedder())

    assert result.updated == 1
    file_a = store.get_file("a.txt")
    assert file_a is not None
    assert file_a.status is FileStatus.FRESH


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


def test_sync_indexes_docx_and_surfaces_heading(store: Store, tmp_path: Path) -> None:
    (tmp_path / "report.docx").write_bytes(make_docx_bytes())

    result = run_sync(store, tmp_path, FakeEmbedder())

    assert result.added == 1
    files = store.list_files()
    assert len(files) == 1
    assert files[0].path == "report.docx"
    assert files[0].status is FileStatus.FRESH

    # The heading chain is written onto the stored chunks (provenance).
    matches = store.dense_search([0.1] * FakeEmbedder().dimension, 10)
    chunks = store.get_chunks([match.rowid for match in matches])
    assert "总体报告 / 第三章" in {chunk.heading for chunk in chunks.values()}

    # And retrieval surfaces a provenance heading on its hits.
    hits = search(store, FakeEmbedder(), "总体", semantic_weight=0.6, lexical_weight=0.4, top_k=5)
    assert hits
    assert any(hit.heading for hit in hits)
