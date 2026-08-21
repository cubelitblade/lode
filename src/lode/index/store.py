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
"""

from __future__ import annotations

import enum
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# The sqlite_vec package ships no type stubs; the ignore is scoped to this
# import line and should be revisited after a dependency upgrade.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]

from lode.embeddings.base import Embedder
from lode.ingestion import Chunk

# Bump when the schema changes incompatibly; a mismatch makes the store
# refuse to open until an explicit rebuild.
SCHEMA_VERSION = 1

# A second process (e.g. CLI next to MCP) waits instead of failing
# immediately with SQLITE_BUSY.
BUSY_TIMEOUT_MS = 5000

# SQLite does not accept parameter binding for DDL, so the VACUUM INTO
# target is a quoted literal; escape any embedded quotes defensively.
_VACUUM_INTO = "VACUUM INTO '{}'"

_TRIGGER_CHUNKS_AI = """
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, chunk_id) VALUES (new.id, new.text, new.chunk_id);
END
"""

_TRIGGER_CHUNKS_AD = """
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id)
    VALUES ('delete', old.id, old.text, old.chunk_id);
END
"""

_TRIGGER_CHUNKS_AU = """
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id)
    VALUES ('delete', old.id, old.text, old.chunk_id);
    INSERT INTO chunks_fts(rowid, text, chunk_id) VALUES (new.id, new.text, new.chunk_id);
END
"""


class StoreError(Exception):
    """Base class for index store failures."""


class SchemaVersionError(StoreError):
    """The existing database was created with an incompatible schema.

    The store refuses to open; an explicit rebuild is required.
    """


class EmbedderUnavailableError(StoreError):
    """The embedder could not provide metadata needed to (re)build the index."""


class ModelStatus(enum.Enum):
    """Result of comparing the stored model with the current embedder."""

    MATCH = "match"
    MISMATCH = "mismatch"
    # Embedder unreachable, so the comparison could not be made. The store
    # stays usable; the search layer decides how to present this.
    UNKNOWN = "unknown"


class FileStatus(enum.StrEnum):
    CURRENT = "current"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class FileRecord:
    """Metadata row for one indexed file (mirrors the `files` table)."""

    path: str
    digest: str
    mtime: float
    size: int
    status: FileStatus = FileStatus.CURRENT


@dataclass(frozen=True, slots=True)
class ChunkWithPath:
    """Chunk content joined with its file metadata (search-layer payload)."""

    chunk_id: str
    text: str
    heading: str
    path: str
    file_status: FileStatus


class Store:
    """SQLite index store: single connection, WAL, lock-protected.

    Construction either opens an existing index or creates a new one.
    Creating a new index needs the vector dimension from the embedder and
    may reach the network; opening an existing, compatible index never
    touches the embedder for metadata (model detection is the only embedder
    access, and it is fault-tolerant).
    """

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        # The workspace may not have a `.lode/` directory yet; the store
        # creates it so the first `lode mine` works on a bare workspace.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._embedder = embedder
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self._stored_model_id: str | None = None
        self._stored_dimension: int | None = None
        self._model_status: ModelStatus = ModelStatus.UNKNOWN

        try:
            self._configure_connection()
            if not self._has_table("meta"):
                self._initialize()
            else:
                self._load_metadata()
                self._detect_model()
        except Exception:
            self._conn.close()
            raise

    # -- construction helpers ------------------------------------------------

    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def _initialize(self) -> None:
        """Create the schema on a brand-new database.

        This is the only path that needs the embedder's dimension, so it is
        also the only path that may hit the network at open time — a fresh
        database has no data to serve yet, so failing here is honest.
        """
        try:
            dimension = self._embedder.dimension
            model_id = self._embedder.model_id
        except Exception as exc:
            raise EmbedderUnavailableError(
                f"cannot create index: could not determine embedding metadata: {exc}"
            ) from exc
        with self._conn:
            self._create_schema(dimension)
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("model_id", model_id)
            self._set_meta("dimension", str(dimension))
        self._stored_model_id = model_id
        self._stored_dimension = dimension
        self._model_status = ModelStatus.MATCH

    def _load_metadata(self) -> None:
        """Read schema version and embedding metadata from an existing database.

        Deliberately does not touch the embedder: the dimension comes from
        `meta`, so an unreachable embedding endpoint cannot block opening.
        """
        version = self._meta_get("schema_version")
        if version != str(SCHEMA_VERSION):
            raise SchemaVersionError(
                f"database schema version {version!r} is incompatible with "
                f"supported version {SCHEMA_VERSION}; run an explicit rebuild"
            )
        dimension = self._meta_get("dimension")
        if dimension is None:
            raise SchemaVersionError("database is missing dimension metadata; run an explicit rebuild")
        self._stored_dimension = int(dimension)
        self._stored_model_id = self._meta_get("model_id")

    def _detect_model(self) -> None:
        """Compare the stored model with the current embedder, fault-tolerantly.

        An unreachable embedder maps to `ModelStatus.UNKNOWN`: the store
        stays open and search can keep serving cached data (PLAN D7).
        """
        stored = self._stored_model_id
        if stored is None:
            self._model_status = ModelStatus.UNKNOWN
            return
        try:
            current = self._embedder.model_id
        except Exception:
            self._model_status = ModelStatus.UNKNOWN
            return
        self._model_status = ModelStatus.MATCH if current == stored else ModelStatus.MISMATCH

    # -- public API ----------------------------------------------------------

    @property
    def model_status(self) -> ModelStatus:
        """How the stored model compares to the current embedder."""
        return self._model_status

    @property
    def dimension(self) -> int:
        """Vector dimension of the stored index (from `meta`, not the embedder)."""
        if self._stored_dimension is None:
            raise StoreError("store has no dimension; was it initialized?")
        return self._stored_dimension

    @property
    def stored_model_id(self) -> str | None:
        """Model identifier recorded in `meta` at (re)build time."""
        return self._stored_model_id

    def replace_file(self, file: FileRecord, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Atomically re-index one file: replace its chunks, vectors, and FTS rows.

        The previous chunks of the file (if any) are deleted first; the FTS
        table is kept in sync by triggers, vectors are removed explicitly.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM files WHERE path = ?", (file.path,)).fetchone()
            now = time.time()
            if row is None:
                cursor = self._conn.execute(
                    "INSERT INTO files (path, digest, mtime, size, status, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (file.path, file.digest, file.mtime, file.size, file.status.value, now),
                )
                file_id = cast(int, cursor.lastrowid)
            else:
                file_id = int(row[0])
                self._conn.execute(
                    "UPDATE files SET digest = ?, mtime = ?, size = ?, status = ?, indexed_at = ? WHERE id = ?",
                    (file.digest, file.mtime, file.size, file.status.value, now, file_id),
                )
                self._delete_chunks(file_id)
            self._insert_chunks(file_id, chunks, vectors)

    def remove_file(self, path: str) -> None:
        """Delete a file and all of its chunks (vectors + FTS rows included)."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
            if row is None:
                return
            file_id = int(row[0])
            self._delete_chunks(file_id)
            self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def mark_stale(self, path: str) -> None:
        """Mark a file's index data as outdated (content changed, sync pending)."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status = ? WHERE path = ?",
                (FileStatus.STALE.value, path),
            )

    # -- query primitives (consumed by the search layer) ---------------------

    def dense_search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        """``(chunk rowid, distance)`` of the k nearest neighbors, ascending."""
        if k <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT rowid, distance FROM chunk_vectors WHERE embedding MATCH ? AND k = {int(k)} ORDER BY distance",
                (json.dumps(vector),),
            ).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """``(chunk rowid, bm25 score)`` of the k best FTS5 matches.

        BM25 scores are negative in SQLite (closer to zero = better), so
        callers should rank by descending score.
        """
        if k <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunks_fts) FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) DESC LIMIT ?",
                (query, k),
            ).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def get_chunks(self, rowids: list[int]) -> dict[int, ChunkWithPath]:
        """Chunk contents for the given rowids, keyed by rowid."""
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.chunk_id, c.text, c.heading, f.path, f.status "
                "FROM chunks c JOIN files f ON f.id = c.file_id "
                f"WHERE c.id IN ({placeholders})",
                [int(r) for r in rowids],
            ).fetchall()
        return {
            int(row[0]): ChunkWithPath(
                chunk_id=row[1], text=row[2], heading=row[3], path=row[4], file_status=FileStatus(row[5])
            )
            for row in rows
        }

    def list_files(self) -> list[FileRecord]:
        """All indexed files, sorted by path."""
        with self._lock:
            rows = self._conn.execute("SELECT path, digest, mtime, size, status FROM files ORDER BY path").fetchall()
        return [_row_to_file(row) for row in rows]

    def get_file(self, path: str) -> FileRecord | None:
        """Metadata for one file, or None if it is not indexed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT path, digest, mtime, size, status FROM files WHERE path = ?", (path,)
            ).fetchone()
        return _row_to_file(row) if row is not None else None

    def rebuild(self) -> None:
        """Drop everything (files included) and re-create the schema.

        The old database is snapshotted to ``<db_path>.bak`` first (via
        ``VACUUM INTO``) so a mistaken rebuild can be recovered manually.
        Requires the embedder to report the dimension; if it is unreachable,
        nothing is changed. Model mismatch is detected separately; callers
        decide when to invoke this.
        """
        with self._lock:
            try:
                dimension = self._embedder.dimension
                model_id = self._embedder.model_id
            except Exception as exc:
                raise EmbedderUnavailableError(
                    f"cannot rebuild index: could not determine embedding metadata: {exc}"
                ) from exc
            backup_path = Path(f"{self._db_path}.bak")
            backup_path.unlink(missing_ok=True)
            quoted = backup_path.as_posix().replace("'", "''")
            self._conn.execute(_VACUUM_INTO.format(quoted))
            with self._conn:
                self._drop_all()
                self._create_schema(dimension)
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                self._set_meta("model_id", model_id)
                self._set_meta("dimension", str(dimension))
            self._stored_model_id = model_id
            self._stored_dimension = dimension
            self._model_status = ModelStatus.MATCH

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    # -- internal write helpers ----------------------------------------------

    def _delete_chunks(self, file_id: int) -> None:
        """Delete all chunks of a file: vectors explicitly, FTS via triggers."""
        chunk_rowids = [
            int(row[0]) for row in self._conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
        ]
        for chunk_rowid in chunk_rowids:
            self._conn.execute("DELETE FROM chunk_vectors WHERE rowid = ?", (chunk_rowid,))
        self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    def _insert_chunks(self, file_id: int, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            cursor = self._conn.execute(
                "INSERT INTO chunks (chunk_id, file_id, seq, text, heading, page) VALUES (?, ?, ?, ?, ?, ?)",
                (chunk.id, file_id, chunk.seq, chunk.text, chunk.heading, chunk.page),
            )
            self._conn.execute(
                "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                (cast(int, cursor.lastrowid), json.dumps(vector)),
            )

    # -- schema ---------------------------------------------------------------

    def _drop_all(self) -> None:
        self._conn.execute("DROP TABLE IF EXISTS chunks_fts")
        self._conn.execute("DROP TABLE IF EXISTS chunk_vectors")
        self._conn.execute("DROP TABLE IF EXISTS chunks")
        self._conn.execute("DROP TABLE IF EXISTS files")
        self._conn.execute("DROP TABLE IF EXISTS meta")

    def _create_schema(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._conn.execute(
            """
            CREATE TABLE meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE files (
                id         INTEGER PRIMARY KEY,
                path       TEXT UNIQUE NOT NULL,
                digest     TEXT NOT NULL,
                mtime      REAL NOT NULL,
                size       INTEGER NOT NULL,
                status     TEXT NOT NULL DEFAULT 'current',
                indexed_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE chunks (
                id       INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                file_id  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                seq      INTEGER NOT NULL,
                text     TEXT NOT NULL,
                heading  TEXT,
                page     INTEGER,
                UNIQUE (file_id, seq)
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE chunk_vectors USING vec0(
                embedding FLOAT[{dimension}]
            )
            """
        )
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                chunk_id UNINDEXED,
                content='chunks',
                content_rowid='id'
            )
            """
        )
        self._conn.execute(_TRIGGER_CHUNKS_AI)
        self._conn.execute(_TRIGGER_CHUNKS_AD)
        self._conn.execute(_TRIGGER_CHUNKS_AU)

    # -- meta helpers ---------------------------------------------------------

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row is not None else None

    def _has_table(self, name: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
        return row is not None


def _row_to_file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        path=row[0],
        digest=row[1],
        mtime=row[2],
        size=row[3],
        status=FileStatus(row[4]),
    )
