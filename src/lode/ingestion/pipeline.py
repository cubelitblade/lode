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

from collections.abc import Callable, Sequence
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
    def missing(self) -> int:
        return len(self.missing_files)

    @property
    def pending(self) -> int:
        """Work for the next ``sync``: new + changed + missing."""
        return self.new + self.changed + self.missing


@dataclass(frozen=True, slots=True)
class FailedFile:
    """A file that failed to (re)index, with the reason it failed.

    Structured so machine-readable output (and a future MCP layer) can read
    ``path`` and ``error`` apart instead of parsing a single string.
    """

    path: str
    error: str


@dataclass(slots=True)
class SyncSummary:
    """Result of a full update pass."""

    added_files: list[str] = field(default_factory=list[str])
    updated_files: list[str] = field(default_factory=list[str])
    removed_files: list[str] = field(default_factory=list[str])
    unchanged: int = 0
    skipped: int = 0
    failed: list[FailedFile] = field(default_factory=list[FailedFile])

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
    """Detect workspace changes against the index and mark changed files stale.

    Read-mostly: only ``files.status`` is written (to STALE), so search can
    flag files whose index may be out of date. No content is embedded or
    replaced here — that is ``sync``'s job.

    Classification is purely a disk-vs-index stat comparison:

    * ``new`` — on disk but not indexed.
    * ``changed`` — mtime/size differ from the index; the file is marked stale.
    * ``unchanged`` — mtime/size match the index. A lingering STALE marker from
      a previous failed run is cleared by ``sync``, not reported here.
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
            summary.unchanged_files.append(rel_text)
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
    *,
    report: Callable[[int, int, str], None] | None = None,
) -> SyncSummary:
    """Update the index to match the workspace: embed what changed, prune the rest.

    A file is re-indexed when it is new, changed (mtime+size differ), or
    still stale from a previous failed run — stale-but-unchanged files must
    be retried so a transient failure does not leave them stale forever.

    ``report`` (optional) is called as ``report(done, total, path)`` before
    each file is processed — ``done`` is the count already finished, ``total``
    the number of discovered files, and ``path`` the file now being handled.
    It lets the CLI surface progress without the pipeline depending on a
    concrete UI library.
    """
    discovered = discover(root, ignore_files)
    indexed = {file.path: file for file in store.list_files()}
    on_disk = {rel.as_posix() for rel in discovered}
    total = len(discovered)

    summary = SyncSummary()
    for idx, rel in enumerate(discovered, start=1):
        rel_text = rel.as_posix()
        path = root / rel
        if report is not None:
            report(idx - 1, total, rel_text)
        if not is_supported(rel.suffix):
            summary.skipped += 1
            continue
        try:
            data = path.read_bytes()
            stat = path.stat()
        except OSError as exc:
            summary.failed.append(FailedFile(path=rel_text, error=f"could not read file: {exc}"))
            store.mark_stale(rel_text)
            continue

        known = indexed.get(rel_text)
        if (
            known is not None
            and known.status is FileStatus.FRESH
            and known.mtime == stat.st_mtime
            and known.size == stat.st_size
        ):
            summary.unchanged += 1
            continue

        digest = file_digest(data)
        record = FileRecord(path=rel_text, digest=digest, mtime=stat.st_mtime, size=stat.st_size)
        if store.reference_file(record):
            # Identical content is already indexed (a copy elsewhere):
            # reuse it instead of re-extracting and re-embedding.
            if known is None:
                summary.added_files.append(rel_text)
            else:
                summary.updated_files.append(rel_text)
            continue

        try:
            segments = extract_document(data, rel.suffix)
        except Exception as exc:
            summary.failed.append(FailedFile(path=rel_text, error=str(exc)))
            store.mark_stale(rel_text)
            continue
        if segments is None:
            summary.skipped += 1
            continue

        try:
            chunks = splitter.split_segments(segments)
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            store.replace_file(record, chunks, vectors)
            if known is None:
                summary.added_files.append(rel_text)
            else:
                summary.updated_files.append(rel_text)
        except Exception as exc:
            # The old snapshot stays queryable; the file is flagged so search
            # can tell the user it may be out of date (PLAN D7).
            summary.failed.append(FailedFile(path=rel_text, error=str(exc)))
            store.mark_stale(rel_text)

    for path in indexed:
        if path not in on_disk:
            store.remove_file(path)
            summary.removed_files.append(path)

    if report is not None:
        report(total, total, "")

    return summary
