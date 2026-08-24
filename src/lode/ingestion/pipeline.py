"""The ingestion pipeline: discovery -> extraction -> chunking -> embedding.

Two explicit actions: ``detect_changes`` (classify and mark stale, never
touches the embedder) and ``sync`` (re-embed the flagged files and prune
removed ones; the only step that needs the embedder). ``sync`` consumes the
buckets ``detect_changes`` computed and never re-classifies. Individual file
failures never abort the run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lode.embeddings.base import Embedder
from lode.index.store import EmbedderUnavailableError, FileRecord, FileStatus, Store, StoreError
from lode.ingestion.digest import file_digest
from lode.ingestion.discover import discover
from lode.ingestion.errors import ExtractionError
from lode.ingestion.extract import extract_document, is_supported
from lode.ingestion.split import SegmentSplitter


@dataclass(slots=True)
class DetectResult:
    """Result of a detection-only pass over the workspace.

    Counts are derived from the per-status path lists, so the summary and the
    ``pending`` listing always agree. ``changed_files`` includes residual stale
    files (stat matches but a previous run failed) so ``sync`` retries them.
    ``stale_paths`` is the side-effect list produced by ``classify`` and applied
    by ``detect_changes``; it is not part of the work buckets.
    """

    unchanged_files: list[str] = field(default_factory=list[str])
    new_files: list[str] = field(default_factory=list[str])
    changed_files: list[str] = field(default_factory=list[str])
    missing_files: list[str] = field(default_factory=list[str])
    skipped: int = 0  # on disk, unsupported format
    stale_paths: list[str] = field(default_factory=list[str])

    @classmethod
    def from_paths(
        cls,
        *,
        unchanged_files: list[str] | None = None,
        new_files: list[str] | None = None,
        changed_files: list[str] | None = None,
        missing_files: list[str] | None = None,
        skipped: int = 0,
        stale_paths: list[str] | None = None,
    ) -> DetectResult:
        """Build a result from explicit path lists (mainly for tests/render)."""
        return cls(
            unchanged_files=unchanged_files or [],
            new_files=new_files or [],
            changed_files=changed_files or [],
            missing_files=missing_files or [],
            skipped=skipped,
            stale_paths=stale_paths or [],
        )

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

    @property
    def dirty(self) -> bool:
        """Whether the library holds data that is out of date.

        ``changed`` (disk differs or residual stale) and ``missing`` (a file
        was removed) both mean the index no longer reflects the workspace.
        ``new`` alone does not — it is an incomplete index, not a dirty one.
        """
        return bool(self.changed_files or self.missing_files)


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


def classify(
    indexed: Mapping[str, FileRecord],
    root: Path,
    ignore_files: Sequence[str] = (),
) -> DetectResult:
    """Classify the workspace against an index snapshot, with no side effects.

    Pure disk-vs-index stat comparison over a ``{path: FileRecord}`` snapshot.
    Returns the ``DetectResult`` buckets plus ``stale_paths`` (paths whose stat
    differs and need their status flipped to STALE). Never touches the store or
    the embedder, so it works with an empty snapshot (no index yet) and with
    the embedding endpoint down.

    * ``new`` — on disk but not indexed.
    * ``changed`` — stat differs (or residual stale marker); added to ``stale_paths``.
    * ``unchanged`` — stat matches and the file is fresh.
    * ``missing`` — indexed but gone from disk; ``sync`` deletes it.
    """
    discovered = discover(root, ignore_files)
    on_disk = {rel.as_posix() for rel in discovered}

    summary = DetectResult()
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
            if known.status is FileStatus.STALE:
                # Residual stale: stat matches but a previous run failed.
                # Fold into changed so sync retries it; the marker is already set.
                summary.changed_files.append(rel_text)
            else:
                summary.unchanged_files.append(rel_text)
        else:
            summary.stale_paths.append(rel_text)
            summary.changed_files.append(rel_text)

    for path in indexed:
        if path not in on_disk:
            summary.missing_files.append(path)

    return summary


def detect_changes(
    store: Store,
    root: Path,
    ignore_files: Sequence[str] = (),
) -> DetectResult:
    """Detect workspace changes and mark changed files stale.

    Thin wrapper over ``classify``: snapshots the index, classifies, then applies
    the ``stale_paths`` side effects (flipping ``files.status`` to STALE). No
    content is embedded or replaced here — that is ``sync``'s job.
    """
    indexed = {file.path: file for file in store.list_files()}
    result = classify(indexed, root, ignore_files)
    for rel_text in result.stale_paths:
        store.mark_stale(rel_text)
    return result


def sync(
    store: Store,
    root: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    *,
    detect: DetectResult,
    report: Callable[[int, int, str], None] | None = None,
) -> SyncSummary:
    """Update the index to match the workspace, consuming a detection result.

    Never classifies: it only works the buckets ``detect`` computed —
    ``new_files`` (added), ``changed_files`` (updated, including residual
    stale), and ``missing_files`` (removed). ``report`` (optional) is called as
    ``report(done, total, path)`` before each file is processed, letting the
    CLI surface progress without depending on a concrete UI library.
    """
    summary = SyncSummary()
    summary.unchanged = len(detect.unchanged_files)
    summary.skipped = detect.skipped

    to_embed = [*detect.new_files, *detect.changed_files]
    total = len(to_embed)
    for idx, rel_text in enumerate(to_embed, start=1):
        path = root / rel_text
        if report is not None:
            report(idx - 1, total, rel_text)
        try:
            data = path.read_bytes()
            stat = path.stat()
        except OSError as exc:
            summary.failed.append(FailedFile(path=rel_text, error=f"could not read file: {exc}"))
            store.mark_stale(rel_text)
            continue

        digest = file_digest(data)
        record = FileRecord(path=rel_text, digest=digest, mtime=stat.st_mtime, size=stat.st_size)
        if store.reference_file(record):
            # Identical content is already indexed (a copy elsewhere):
            # reuse it instead of re-extracting and re-embedding.
            if rel_text in detect.new_files:
                summary.added_files.append(rel_text)
            else:
                summary.updated_files.append(rel_text)
            continue

        try:
            segments = extract_document(data, Path(rel_text).suffix)
        except ExtractionError as exc:
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
            if rel_text in detect.new_files:
                summary.added_files.append(rel_text)
            else:
                summary.updated_files.append(rel_text)
        except (EmbedderUnavailableError, StoreError) as exc:
            # The old snapshot stays queryable; the file is flagged so search
            # can tell the user it may be out of date. Only *domain* errors
            # are caught here — a programming error (e.g. a bug in the
            # splitter) must propagate instead of being silently downgraded
            # to a per-file failure.
            summary.failed.append(FailedFile(path=rel_text, error=str(exc)))
            store.mark_stale(rel_text)

    for rel_text in detect.missing_files:
        store.remove_file(rel_text)
        summary.removed_files.append(rel_text)

    if report is not None:
        report(total, total, "")

    return summary
