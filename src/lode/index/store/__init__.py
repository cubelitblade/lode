"""SQLite index store: files + chunks + vectors + FTS in one database.

Owns the single SQLite connection, the vec0 virtual table (dense vectors),
the FTS5 external-content table (sparse/BM25), and the triggers that keep
the FTS index in sync with `chunks`. Layout: `errors` (exceptions),
`records` (enums, dataclasses, row mapping), `schema` (DDL + triggers),
`core` (the `Store` class).
"""

from __future__ import annotations

from lode.index.store.core import BUSY_TIMEOUT_MS, Store
from lode.index.store.errors import (
    DimensionMismatchError,
    EmbedderUnavailableError,
    MissingEmbedderError,
    SchemaVersionError,
    StoreError,
)
from lode.index.store.records import (
    ChunkWithPath,
    DenseMatch,
    FileRecord,
    FileStatus,
    ModelStatus,
    PathRef,
    SparseMatch,
)
from lode.index.store.schema import SCHEMA_VERSION

__all__ = [
    "BUSY_TIMEOUT_MS",
    "SCHEMA_VERSION",
    "ChunkWithPath",
    "DenseMatch",
    "DimensionMismatchError",
    "EmbedderUnavailableError",
    "FileRecord",
    "FileStatus",
    "MissingEmbedderError",
    "ModelStatus",
    "PathRef",
    "SchemaVersionError",
    "SparseMatch",
    "Store",
    "StoreError",
]
