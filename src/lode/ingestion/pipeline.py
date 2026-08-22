"""The ingestion pipeline: discovery -> extraction -> chunking -> embedding.

Two explicit actions, mirroring PLAN D7 (detection and update are separate):

* ``survey_workspace`` — detect changes and mark files stale. Never touches
  the embedder, so it works even when the embedding endpoint is down.
* ``sync`` — the actual update: re-embed changed/stale files and clean up
  files that disappeared. This is the only step that requires the embedder.

Individual file failures never abort the run: the file is marked stale (or
reported for new files) and the rest of the workspace continues.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lode.embeddings.base import Embedder
from lode.index.store import FileRecord, FileStatus, Store
from lode.ingestion.digest import file_digest
from lode.ingestion.discover import discover
from lode.ingestion.extract import extract_document, is_supported
from lode.ingestion.split import SegmentSplitter


@dataclass(slots=True)
class SurveySummary:
    """Result of a detection-only pass over the workspace.

    Counts are derived from the per-status path lists, so the summary line
    and the ``pending`` listing always agree.
    """

    unchanged_files: list[str] = field(default_factory=list[str])
    new_files: list[str] = field(default_factory=list[str])
    changed_files: list[str] = field(default_factory=list[str])
    stale_files: list[str] = field(default_factory=list[str])
    missing_files: list[str] = field(default_factory=list[str])
    skipped: int = 0  # on disk, unsupported format

    @property
    def unchanged(self) -> int:
        return len(self.unchanged_files)

    @property
    def new(self) -> int:
        return len(self.new_files)

    @property
    def changed(self) -> int:
        return len(self.changed_files)

    @property
    def stale(self) -> int:
        return len(self.stale_files)

    @property
    def missing(self) -> int:
        return len(self.missing_files)

    @property
    def pending(self) -> int:
        """Work for the next ``sync``: new + changed + stale + missing."""
        return self.new + self.changed + self.stale + self.missing


@dataclass(slots=True)
class SyncSummary:
    """Result of a full update pass."""

    added_files: list[str] = field(default_factory=list[str])
    updated_files: list[str] = field(default_factory=list[str])
    removed_files: list[str] = field(default_factory=list[str])
    unchanged: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list[str])

    @property
    def added(self) -> int:
        return len(self.added_files)

    @property
    def updated(self) -> int:
        return len(self.updated_files)

    @property
    def removed(self) -> int:
        return len(self.removed_files)


def survey_workspace(
    store: Store,
    root: Path,
    ignore_files: Sequence[str] = (),
) -> SurveySummary:
    """Detect workspace changes against the index and mark stale files.

    Read-mostly: only ``files.status`` is written (to STALE). No content is
    embedded or replaced here — that is ``sync``'s job.
    """
    discovered = discover(root, ignore_files)
    indexed = {file.path: file for file in store.list_files()}
    on_disk = {rel.as_posix() for rel in discovered}

    summary = SurveySummary()
    for rel in discovered:
        rel_text = rel.as_posix()
        if not is_supported(rel.suffix):
            summary.skipped += 1
            continue
        stat = (root / rel).stat()
        known = indexed.get(rel_text)
        if known is None:
            summary.new_files.append(rel_text)
        elif known.mtime == stat.st_mtime and known.size == stat.st_size:
            if known.status is FileStatus.CURRENT:
                summary.unchanged_files.append(rel_text)
            else:
                summary.stale_files.append(rel_text)
        else:
            store.mark_stale(rel_text)
            summary.changed_files.append(rel_text)

    for path in indexed:
        if path not in on_disk:
            summary.missing_files.append(path)

    return summary


def sync(
    store: Store,
    root: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    ignore_files: Sequence[str] = (),
) -> SyncSummary:
    """Update the index to match the workspace: embed what changed, prune the rest.

    A file is re-indexed when it is new, changed (mtime+size differ), or
    still stale from a previous failed run — stale-but-unchanged files must
    be retried so a transient failure does not leave them stale forever.
    """
    discovered = discover(root, ignore_files)
    indexed = {file.path: file for file in store.list_files()}
    on_disk = {rel.as_posix() for rel in discovered}

    summary = SyncSummary()
    for rel in discovered:
        rel_text = rel.as_posix()
        path = root / rel
        if not is_supported(rel.suffix):
            summary.skipped += 1
            continue
        try:
            data = path.read_bytes()
            stat = path.stat()
        except OSError:
            summary.failed.append(rel_text)
            store.mark_stale(rel_text)
            continue

        known = indexed.get(rel_text)
        if (
            known is not None
            and known.status is FileStatus.CURRENT
            and known.mtime == stat.st_mtime
            and known.size == stat.st_size
        ):
            summary.unchanged += 1
            continue

        try:
            segments = extract_document(data, rel.suffix)
        except Exception as exc:
            summary.failed.append(f"{rel_text}: {exc}")
            store.mark_stale(rel_text)
            continue
        if segments is None:
            summary.skipped += 1
            continue

        try:
            chunks = splitter.split_segments(segments)
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            store.replace_file(
                FileRecord(path=rel_text, digest=file_digest(data), mtime=stat.st_mtime, size=stat.st_size),
                chunks,
                vectors,
            )
            if known is None:
                summary.added_files.append(rel_text)
            else:
                summary.updated_files.append(rel_text)
        except Exception as exc:
            # The old snapshot stays queryable; the file is flagged so search
            # can tell the user it may be out of date (PLAN D7).
            summary.failed.append(f"{rel_text}: {exc}")
            store.mark_stale(rel_text)

    for path in indexed:
        if path not in on_disk:
            store.remove_file(path)
            summary.removed_files.append(path)

    return summary
