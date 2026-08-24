"""SQLite index store: files + chunks + vectors + FTS in one database.

The store is the persistence layer under the whole pipeline (PLAN §7).
It owns the single SQLite connection, the vec0 virtual table (dense
vectors), the FTS5 external-content table (sparse/BM25), and the triggers
that keep the FTS index in sync with `chunks`.

Design decisions locked here (confirmed against PLAN D2/D6/D7):

* ``chunks.id`` (INTEGER PRIMARY KEY) is the single rowid shared by the
  FTS5 table (``content_rowid='id'``) and the vec0 table (``rowid``), so
  all three tables join on one key.
* Single connection + RLock: the connection is not thread-safe, so every
  operation takes the lock — reads included. This keeps the store safe to
  serve MCP worker threads later.
* WAL + ``busy_timeout``: concurrent processes (CLI and MCP) can share the
  database file without instant SQLITE_BUSY failures.
* The embedder is only touched to learn the vector dimension when the
  database does not exist yet, and for model detection — which is
  fault-tolerant. An unreachable embedder never prevents the store from
  opening (PLAN D7: search must keep working without the embedding
  endpoint).
* Schema version and embedding metadata (model_id + dimension) live in
  `meta`. A schema mismatch raises `SchemaVersionError` at open and
  requires an explicit rebuild — never an automatic one, and never a
  network-triggered failure.
* ``replace_file`` is the atomic unit of a file's re-indexing.
* ``rebuild()`` drops everything (files included) and re-creates the
  schema after snapshotting the old database to ``<db>.bak`` via
  ``VACUUM INTO``.

Layout: ``errors`` (exception hierarchy), ``records`` (enums, dataclasses,
row mapping), ``schema`` (DDL + triggers), ``store`` (the ``Store`` class).
"""

from __future__ import annotations

from lode.index.store.errors import (
    DimensionMismatchError,
    EmbedderUnavailableError,
    SchemaVersionError,
    StoreError,
)
from lode.index.store.records import (
    ChunkWithPath,
    FileRecord,
    FileStatus,
    ModelStatus,
)
from lode.index.store.schema import SCHEMA_VERSION
from lode.index.store.store import BUSY_TIMEOUT_MS, Store

__all__ = [
    "BUSY_TIMEOUT_MS",
    "SCHEMA_VERSION",
    "ChunkWithPath",
    "DimensionMismatchError",
    "EmbedderUnavailableError",
    "FileRecord",
    "FileStatus",
    "ModelStatus",
    "SchemaVersionError",
    "Store",
    "StoreError",
]
