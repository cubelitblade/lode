"""Data records shared across the index layer: enums, row dataclasses, row mapping."""

from __future__ import annotations

import enum
import sqlite3
from dataclasses import dataclass
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
    """One indexed path joined with its content metadata (mirrors `files` ⋈ `contents`)."""

    path: str
    digest: str
    mtime: float
    size: int
    status: FileStatus = FileStatus.FRESH


@dataclass(frozen=True, slots=True)
class ChunkWithPath:
    """Chunk content joined with representative path metadata (search-layer payload)."""

    digest: str
    text: str
    heading: str
    path: str
    file_status: FileStatus
    page: int | None = None
    seq: int | None = None


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
        path=row[0],
        digest=row[1],
        mtime=row[2],
        size=row[3],
        status=FileStatus(row[4]),
    )


def row_to_chunk_path(row: tuple[Any, ...]) -> ChunkWithPath:
    # Every caller selects c.seq as the 8th column (the store connection uses
    # the default row factory, so rows are plain tuples, not sqlite3.Row).
    return ChunkWithPath(
        digest=row[1],
        text=row[2],
        heading=row[3],
        path=row[4],
        file_status=FileStatus(row[5]),
        page=row[6],
        seq=row[7],
    )
