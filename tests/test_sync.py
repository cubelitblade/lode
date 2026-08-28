"""Tests for the ingestion sync pipeline (index update).

Hermetic: real temp files + FakeEmbedder, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lode.index import FileStatus, Store
from lode.index.ranking import LinearFusion, MinmaxNorm, RetrievalPlan
from lode.index.search import search
from lode.ingestion import Chunk, Segment
from lode.ingestion.pipeline import FailedFile, detect_changes, sync
from lode.ingestion.split import SegmentSplitter
from tests.conftest import SPLITTER, run_sync, write
from tests.fakes import FailingEmbedder, FakeEmbedder, make_docx_bytes


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


def test_detect_and_sync_report_exact_content_rename(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())

    path.rename(tmp_path / "b.txt")
    detect = detect_changes(store, tmp_path)

    # Identical content under a new path is a rename, not new + missing.
    assert detect.renamed_files == [("a.txt", "b.txt")]
    assert detect.new_files == []
    assert detect.missing_files == []
    assert detect.pending == 1
    assert detect.dirty is False

    class CountingEmbedder(FakeEmbedder):
        embed_calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            CountingEmbedder.embed_calls += 1
            return super().embed_documents(texts)

    result = sync(store, tmp_path, CountingEmbedder(), SPLITTER, detect=detect)

    assert result.renamed_files == [("a.txt", "b.txt")]
    assert result.added == 0
    assert result.updated == 0
    assert result.removed == 0
    # Zero embedding cost: the indexed content is re-pointed, not rebuilt.
    assert CountingEmbedder.embed_calls == 0
    files = {file.path: file for file in store.list_files()}
    assert set(files) == {"b.txt"}
    assert all(file.status is FileStatus.FRESH for file in files.values())


def test_rename_pairing_leaves_unmatched_paths_in_place(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())

    # a.txt is gone; b.txt is an identical copy (rename), c.txt is brand-new.
    path.unlink()
    write(tmp_path, "b.txt", "hello world content")
    write(tmp_path, "c.txt", "something else entirely")

    detect = detect_changes(store, tmp_path)
    assert detect.renamed_files == [("a.txt", "b.txt")]
    assert detect.new_files == ["c.txt"]
    assert detect.missing_files == []

    result = sync(store, tmp_path, FakeEmbedder(), SPLITTER, detect=detect)
    assert [pair for pair in result.renamed_files] == [("a.txt", "b.txt")]
    assert result.added_files == ["c.txt"]
    assert result.removed == 0
    assert {file.path for file in store.list_files()} == {"b.txt", "c.txt"}


def test_changed_file_is_never_paired_as_rename(store: Store, tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello world content")
    run_sync(store, tmp_path, FakeEmbedder())

    # Same path, different content: changed, not renamed.
    path.write_text("totally different text now")
    detect = detect_changes(store, tmp_path)
    assert detect.renamed_files == []
    assert detect.changed_files == ["a.txt"]


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
    plan = RetrievalPlan(norm=MinmaxNorm(), fusion=LinearFusion(weights={"semantic": 0.6, "lexical": 0.4}))
    hits = search(store, FakeEmbedder(), "总体", plan=plan, top_k=5)
    assert hits
    assert any(hit.heading for hit in hits)
