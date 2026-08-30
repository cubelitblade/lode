"""Data records shared across the index layer: enums, row dataclasses, row mapping."""

from __future__ import annotations

import enum
import sqlite3
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


class ModelStatus(enum.Enum):
    """Result of comparing the stored model with the current embedder."""

    MATCH = "match"
    MISMATCH = "mismatch"
    # Embedder unreachable, so the comparison could not be made. The store
    # stays usable; the search layer decides how to present this.
    UNKNOWN = "unknown"


class FileStatus(enum.StrEnum):
    FRESH = "fresh"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One indexed path joined with its content metadata (mirrors `files` ⋈ `contents`).

    ``path`` is a workspace-relative :class:`~pathlib.PurePosixPath`: the
    platform-independent domain type (see ``lode.relpath``).
    """

    path: PurePosixPath
    digest: str
    mtime: float
    size: int
    status: FileStatus = FileStatus.FRESH


@dataclass(frozen=True, slots=True)
class PathRef:
    """One workspace path referencing a shared content, with its freshness."""

    path: PurePosixPath
    status: FileStatus


@dataclass(frozen=True, slots=True)
class ChunkWithPath:
    """Chunk content joined with every path referencing it.

    Content is shared by identical files, so a chunk can belong to several
    paths at once; ``primary`` picks the representative one for display.
    """

    digest: str
    text: str
    heading: str
    refs: tuple[PathRef, ...]
    page: int | None = None
    seq: int | None = None

    @property
    def primary(self) -> PathRef:
        """Representative reference: smallest fresh path, else smallest overall."""
        fresh = [ref for ref in self.refs if ref.status is FileStatus.FRESH]
        return min(fresh or self.refs, key=lambda ref: ref.path)


@dataclass(frozen=True, slots=True)
class DenseMatch:
    """One kNN hit: chunk rowid with its L2 distance, ordered nearest first."""

    rowid: int
    distance: float


@dataclass(frozen=True, slots=True)
class SparseMatch:
    """One BM25 hit: chunk rowid with its score, ordered best first.

    SQLite BM25 scores are negative and closer to zero means better, so
    best-first is descending score.
    """

    rowid: int
    score: float


def row_to_file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        path=PurePosixPath(row[0]),
        digest=row[1],
        mtime=row[2],
        size=row[3],
        status=FileStatus(row[4]),
    )


def chunk_from_rows(rows: list[tuple[Any, ...]]) -> ChunkWithPath:
    """Build one chunk from its join rows (one per referencing path).

    Callers select c.seq as the 8th column; the connection uses the default
    row factory, so rows are plain tuples, not sqlite3.Row.
    """
    head = rows[0]
    refs = tuple(
        PathRef(path=PurePosixPath(path), status=FileStatus(status))
        for path, status in sorted({(str(row[4]), str(row[5])) for row in rows})
    )
    return ChunkWithPath(
        digest=head[1],
        text=head[2],
        heading=head[3],
        refs=refs,
        page=head[6],
        seq=head[7],
    )
