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
from pathlib import Path, PurePosixPath

from lode.embeddings.base import Embedder
from lode.index import EmbedderUnavailableError, FileRecord, FileStatus, Store, StoreError
from lode.ingestion.digest import file_digest
from lode.ingestion.discover import discover
from lode.ingestion.errors import ExtractionError
from lode.ingestion.extract import extract_document, is_supported
from lode.ingestion.split import SegmentSplitter
from lode.relpath import to_rel


@dataclass(slots=True)
class DetectResult:
    """Result of a detection-only pass over the workspace.

    Counts are derived from the per-status path lists, so the summary and the
    ``pending`` listing always agree. ``changed_files`` includes residual stale
    files (stat matches but a previous run failed) so ``sync`` retries them.
    ``stale_paths`` is the side-effect list produced by ``classify`` and applied
    by ``detect_changes``; it is not part of the work buckets.
    """

    unchanged_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    new_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    changed_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    missing_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    # ``(old, new)`` pairs whose content digests match exactly; both paths are
    # excluded from ``new_files``/``missing_files`` so buckets never overlap.
    renamed_files: list[tuple[PurePosixPath, PurePosixPath]] = field(
        default_factory=list[tuple[PurePosixPath, PurePosixPath]]
    )
    skipped: int = 0  # on disk, unsupported format
    stale_paths: list[PurePosixPath] = field(default_factory=list[PurePosixPath])

    @classmethod
    def from_paths(
        cls,
        *,
        unchanged_files: Sequence[str | PurePosixPath] | None = None,
        new_files: Sequence[str | PurePosixPath] | None = None,
        changed_files: Sequence[str | PurePosixPath] | None = None,
        missing_files: Sequence[str | PurePosixPath] | None = None,
        renamed_files: Sequence[tuple[str | PurePosixPath, str | PurePosixPath]] | None = None,
        skipped: int = 0,
        stale_paths: Sequence[str | PurePosixPath] | None = None,
    ) -> DetectResult:
        """Build a result from explicit path lists (mainly for tests/render)."""
        return cls(
            unchanged_files=[to_rel(path) for path in unchanged_files or ()],
            new_files=[to_rel(path) for path in new_files or ()],
            changed_files=[to_rel(path) for path in changed_files or ()],
            missing_files=[to_rel(path) for path in missing_files or ()],
            renamed_files=[(to_rel(old), to_rel(new)) for old, new in renamed_files or ()],
            skipped=skipped,
            stale_paths=[to_rel(path) for path in stale_paths or ()],
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
    def renamed(self) -> int:
        return len(self.renamed_files)

    @property
    def pending(self) -> int:
        """Work for the next ``sync``: new + changed + missing + renamed."""
        return self.new + self.changed + self.missing + self.renamed

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

    path: PurePosixPath
    error: str


@dataclass(slots=True)
class SyncSummary:
    """Result of a full update pass."""

    added_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    updated_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    removed_files: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    renamed_files: list[tuple[PurePosixPath, PurePosixPath]] = field(
        default_factory=list[tuple[PurePosixPath, PurePosixPath]]
    )
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

    @property
    def renamed(self) -> int:
        return len(self.renamed_files)


def classify(
    indexed: Mapping[PurePosixPath, FileRecord],
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
    * ``renamed`` — a ``new`` path whose content digest equals an indexed
      ``missing`` path; reported as ``(old, new)`` and excluded from both
      buckets. Pairing is 1:1 and deterministic (paths sorted); leftovers
      stay in their original buckets.
    """
    discovered = discover(root, ignore_files)
    on_disk = set(discovered)

    summary = DetectResult()
    for rel in discovered:
        if not is_supported(rel.suffix):
            summary.skipped += 1
            continue
        stat = (root / str(rel)).stat()
        known = indexed.get(rel)
        if known is None:
            summary.new_files.append(rel)
        elif known.mtime == stat.st_mtime and known.size == stat.st_size:
            if known.status is FileStatus.STALE:
                # Residual stale: stat matches but a previous run failed.
                # Fold into changed so sync retries it; the marker is already set.
                summary.changed_files.append(rel)
            else:
                summary.unchanged_files.append(rel)
        else:
            summary.stale_paths.append(rel)
            summary.changed_files.append(rel)

    for path in indexed:
        if path not in on_disk:
            summary.missing_files.append(path)

    _pair_renames(summary, indexed, root)
    return summary


def _digest_if_readable(path: Path) -> str | None:
    """Content digest of a file, or ``None`` when it cannot be read."""
    try:
        return file_digest(path.read_bytes())
    except OSError:
        return None


def _pair_renames(
    summary: DetectResult,
    indexed: Mapping[PurePosixPath, FileRecord],
    root: Path,
) -> None:
    """Fold exact-content moves out of new/missing into renamed pairs.

    Only the new files are read (one digest each); the missing side already
    carries its digest in the index snapshot. A new file whose digest matches
    a missing path is a move: the content is already indexed, so ``sync`` can
    re-point it with :meth:`Store.reference_file` at zero embedding cost.
    """
    if not summary.new_files or not summary.missing_files:
        return

    missing_by_digest: dict[str, list[PurePosixPath]] = {}
    for path in sorted(summary.missing_files):
        missing_by_digest.setdefault(indexed[path].digest, []).append(path)

    paired: list[tuple[PurePosixPath, PurePosixPath]] = []
    kept_new: list[PurePosixPath] = []
    consumed: set[PurePosixPath] = set()
    for rel in sorted(summary.new_files):
        old: PurePosixPath | None = None
        digest = _digest_if_readable(root / str(rel))
        if digest is not None:
            candidates = missing_by_digest.get(digest, [])
            if candidates:
                old = candidates.pop(0)
        if old is None:
            kept_new.append(rel)
        else:
            consumed.add(old)
            paired.append((old, rel))

    if paired:
        summary.renamed_files = paired
        summary.new_files = kept_new
        summary.missing_files = [path for path in summary.missing_files if path not in consumed]


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


def _embed_one_file(
    store: Store,
    root: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    summary: SyncSummary,
    *,
    rel: PurePosixPath,
    added_paths: frozenset[PurePosixPath],
) -> None:
    """Embed a single file into the index, updating ``summary`` in place.

    The per-file flow: read -> content-reuse check -> extract -> split+embed+
    replace. ``added_paths`` distinguishes additions from updates (a rename
    fallback counts as an addition even though it is not in ``new_files``).
    Domain failures (unreadable file, extraction error, embedder/store error)
    are recorded on ``summary`` and the file is marked stale so a later run
    retries it; a programming error (e.g. a bug in the splitter) propagates
    instead of being silently downgraded to a per-file failure.
    """
    path = root / str(rel)
    try:
        data = path.read_bytes()
        stat = path.stat()
    except OSError as exc:
        summary.failed.append(FailedFile(path=rel, error=f"could not read file: {exc}"))
        store.mark_stale(rel)
        return

    digest = file_digest(data)
    record = FileRecord(path=rel, digest=digest, mtime=stat.st_mtime, size=stat.st_size)
    if store.reference_file(record):
        # Identical content is already indexed (a copy elsewhere):
        # reuse it instead of re-extracting and re-embedding.
        if rel in added_paths:
            summary.added_files.append(rel)
        else:
            summary.updated_files.append(rel)
        return

    try:
        segments = extract_document(data, rel.suffix)
    except ExtractionError as exc:
        summary.failed.append(FailedFile(path=rel, error=str(exc)))
        store.mark_stale(rel)
        return
    if segments is None:
        summary.skipped += 1
        return

    try:
        chunks = splitter.split_segments(segments)
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store.replace_file(record, chunks, vectors)
        if rel in added_paths:
            summary.added_files.append(rel)
        else:
            summary.updated_files.append(rel)
    except (EmbedderUnavailableError, StoreError) as exc:
        # The old snapshot stays queryable; the file is flagged so search
        # can tell the user it may be out of date. Only *domain* errors are
        # caught here — a programming error (e.g. a bug in the splitter)
        # must propagate instead of being silently downgraded to a per-file
        # failure.
        summary.failed.append(FailedFile(path=rel, error=str(exc)))
        store.mark_stale(rel)


def sync(
    store: Store,
    root: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    *,
    detect: DetectResult,
    report: Callable[[int, int, PurePosixPath | None], None] | None = None,
) -> SyncSummary:
    """Update the index to match the workspace, consuming a detection result.

    Never classifies: it only works the buckets ``detect`` computed —
    ``renamed_files`` (re-pointed at zero embedding cost), ``new_files``
    (added), ``changed_files`` (updated, including residual stale), and
    ``missing_files`` (removed). ``report`` (optional) is called as
    ``report(done, total, path)`` before each embedded file is processed,
    letting the CLI surface progress without depending on a concrete UI
    library.
    """
    summary = SyncSummary()
    summary.unchanged = len(detect.unchanged_files)
    summary.skipped = detect.skipped

    # Renames first: they are pure re-pointings and never need the embedder.
    # A digest that no longer resolves (snapshot changed under us) falls back
    # to a full embed for the new path plus pruning of the old one.
    rename_fallbacks: list[PurePosixPath] = []
    fallback_removals: list[PurePosixPath] = []
    for old, new in detect.renamed_files:
        try:
            data = (root / str(new)).read_bytes()
            stat = (root / str(new)).stat()
        except OSError as exc:
            summary.failed.append(FailedFile(path=new, error=f"could not read file: {exc}"))
            store.remove_file(old)
            continue
        record = FileRecord(path=new, digest=file_digest(data), mtime=stat.st_mtime, size=stat.st_size)
        if store.reference_file(record):
            summary.renamed_files.append((old, new))
            # Drop the old path reference only after the new one claims the
            # content, so the shared content can never be GC'd in between.
            store.remove_file(old)
        else:
            rename_fallbacks.append(new)
            fallback_removals.append(old)

    to_embed = [*detect.new_files, *rename_fallbacks, *detect.changed_files]
    # Rename fallbacks were never indexed under their new path, so they count
    # as additions even though they are not in ``detect.new_files``.
    added_paths = frozenset([*detect.new_files, *rename_fallbacks])
    total = len(to_embed)
    for idx, rel in enumerate(to_embed, start=1):
        if report is not None:
            report(idx - 1, total, rel)
        _embed_one_file(
            store,
            root,
            embedder,
            splitter,
            summary,
            rel=rel,
            added_paths=added_paths,
        )

    for rel in [*detect.missing_files, *fallback_removals]:
        store.remove_file(rel)
        summary.removed_files.append(rel)

    if report is not None:
        # Completion report: ``None`` path means the run is done (the CLI
        # shows a neutral description; an empty PurePosixPath would render
        # as "." because paths are always truthy).
        report(total, total, None)

    return summary
