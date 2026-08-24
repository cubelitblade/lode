"""The ``Store`` class: lifecycle, metadata, writes, and query primitives."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

# The sqlite_vec package ships no type stubs; the ignore is scoped to this
# import line and should be revisited after a dependency upgrade.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]

from lode.embeddings.base import Embedder
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
    SparseMatch,
    chunk_from_rows,
    row_to_file,
)
from lode.index.store.schema import SCHEMA_VERSION, create_schema, drop_all
from lode.ingestion import Chunk

BUSY_TIMEOUT_MS = 5000

# SQLite does not accept parameter binding for DDL, so the VACUUM INTO
# target is a quoted literal; escape any embedded quotes defensively.
_VACUUM_INTO = "VACUUM INTO '{}'"

# Shared column list for chunk-with-path queries; `row_to_chunk_path` reads
# these positions positionally.
_CHUNK_COLUMNS = "c.id, c.digest, c.text, c.heading, f.path, f.status, c.page, c.seq"


class Store:
    """SQLite index store: single connection, WAL, lock-protected.

    Creating a new index needs an embedder (for the vector dimension) and may
    reach the network; opening an existing, compatible index never touches the
    embedder unless its model status is queried.
    """

    def __init__(self, db_path: Path, embedder: Embedder | None = None) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._embedder = embedder
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self._stored_model_id: str | None = None
        self._stored_dimension: int | None = None
        self._model_status = ModelStatus.UNKNOWN
        self._model_checked = False

        try:
            self._configure_connection()
            if not self._has_table("meta"):
                self._initialize()
            else:
                self._load_metadata()
        except Exception:
            self._conn.close()
            raise

    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Digest prefixes are matched with LIKE; every stored digest is
        # lowercase hex, so case-sensitive matching both preserves that
        # contract and lets prefix lookups use idx_chunks_digest.
        self._conn.execute("PRAGMA case_sensitive_like=ON")
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def _require_embedder(self) -> Embedder:
        if self._embedder is None:
            raise MissingEmbedderError(
                "creating or rebuilding the index requires an embedder; "
                "open an existing index instead, or configure one"
            )
        return self._embedder

    def _initialize(self) -> None:
        """Create the schema on a brand-new database.

        The only path that needs the embedder's dimension — and thus the
        only one that may reach the network at open time; a fresh database
        has no data to serve yet, so failing here is honest.
        """
        embedder = self._require_embedder()
        try:
            dimension = embedder.dimension
            model_id = embedder.model_id
        except Exception as exc:
            raise EmbedderUnavailableError(
                f"cannot create index: could not determine embedding metadata: {exc}"
            ) from exc
        with self._conn:
            create_schema(self._conn, dimension)
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("model_id", model_id)
            self._set_meta("dimension", str(dimension))
        self._stored_model_id = model_id
        self._stored_dimension = dimension
        self._model_status = ModelStatus.MATCH
        self._model_checked = True

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
        stays open and search can keep serving cached data.
        """
        self._model_checked = True
        stored = self._stored_model_id
        if stored is None or self._embedder is None:
            self._model_status = ModelStatus.UNKNOWN
            return
        try:
            current = self._embedder.model_id
        except Exception:
            self._model_status = ModelStatus.UNKNOWN
            return
        self._model_status = ModelStatus.MATCH if current == stored else ModelStatus.MISMATCH

    @property
    def model_status(self) -> ModelStatus:
        """How the stored model compares to the current embedder.

        Resolved lazily on first access, so merely opening the store never
        hits the embedding endpoint.
        """
        if not self._model_checked:
            self._detect_model()
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
        """Atomically point ``file.path`` at the content addressed by ``file.digest``.

        The content is created together with its chunks, vectors, and FTS
        rows when this is its first reference; an already-present content is
        reused untouched (content addressing makes its chunks identical).
        When the path moves away from a previous content, that content is
        dropped once its last reference disappears.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        with self._lock, self._conn:
            row = self._conn.execute("SELECT content_id FROM files WHERE path = ?", (file.path,)).fetchone()
            previous_content_id = int(row[0]) if row is not None else None
            content_id, created = self._ensure_content(file.digest)
            self._conn.execute(
                """
                INSERT INTO files (path, content_id, mtime, size, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_id = excluded.content_id,
                    mtime = excluded.mtime,
                    size = excluded.size,
                    status = excluded.status
                """,
                (file.path, content_id, file.mtime, file.size, file.status.value),
            )
            if created:
                self._insert_chunks(content_id, chunks, vectors)
            if previous_content_id is not None and previous_content_id != content_id:
                self._gc_content_if_orphaned(previous_content_id)

    def reference_file(self, file: FileRecord) -> bool:
        """Point ``file.path`` at already-indexed content; report whether it existed.

        Unlike :meth:`replace_file`, no chunks are written — the caller
        claims the content addressed by ``file.digest`` is already indexed.
        When it is not, nothing changes and False is returned.
        """
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM contents WHERE digest = ?", (file.digest,)).fetchone()
            if row is None:
                return False
            content_id = int(row[0])
            previous = self._conn.execute("SELECT content_id FROM files WHERE path = ?", (file.path,)).fetchone()
            self._conn.execute(
                """
                INSERT INTO files (path, content_id, mtime, size, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_id = excluded.content_id,
                    mtime = excluded.mtime,
                    size = excluded.size,
                    status = excluded.status
                """,
                (file.path, content_id, file.mtime, file.size, file.status.value),
            )
            if previous is not None and int(previous[0]) != content_id:
                self._gc_content_if_orphaned(int(previous[0]))
            return True

    def remove_file(self, path: str) -> None:
        """Delete a path reference; drop its content when this was the last one."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT content_id FROM files WHERE path = ?", (path,)).fetchone()
            if row is None:
                return
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
            self._gc_content_if_orphaned(int(row[0]))

    def mark_stale(self, path: str) -> None:
        """Mark a path's snapshot as outdated (content changed, sync pending).

        Unknown paths are ignored: a new file that fails before its first
        successful indexing leaves no row behind.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status = ? WHERE path = ?",
                (FileStatus.STALE.value, path),
            )

    def content_id_for(self, digest: str) -> int | None:
        """Rowid of the content with this exact digest, or None when absent."""
        with self._lock:
            row = self._conn.execute("SELECT id FROM contents WHERE digest = ?", (digest,)).fetchone()
        return int(row[0]) if row is not None else None

    def dense_search(self, vector: list[float], k: int) -> list[DenseMatch]:
        """k nearest neighbors as ``(rowid, distance)``, nearest first."""
        if k <= 0:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    # vec0's MATCH syntax takes the limit as a literal, not a
                    # bound parameter; int() keeps the interpolation safe.
                    f"SELECT rowid, distance FROM chunk_vectors "
                    f"WHERE embedding MATCH ? AND k = {int(k)} ORDER BY distance",
                    (json.dumps(vector),),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "Dimension mismatch" in str(exc):
                    raise DimensionMismatchError(self.dimension, len(vector), operation="query") from exc
                raise
        return [DenseMatch(int(row[0]), float(row[1])) for row in rows]

    def sparse_search(self, query: str, k: int) -> list[SparseMatch]:
        """k best FTS5 matches for ``query``, best (closest-to-zero BM25) first."""
        if k <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunks_fts) FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) DESC LIMIT ?",
                (query, k),
            ).fetchall()
        return [SparseMatch(int(row[0]), float(row[1])) for row in rows]

    def get_chunks(self, rowids: list[int]) -> dict[int, ChunkWithPath]:
        """Chunk contents for the given rowids, keyed by rowid.

        Each chunk carries every path referencing its content.
        """
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN files f ON f.content_id = c.content_id "
                f"WHERE c.id IN ({placeholders})",
                [int(r) for r in rowids],
            ).fetchall()
        return {int(group[0][0]): chunk_from_rows(group) for group in _group_rows(rows).values()}

    def find_chunks_by_digest(self, prefix: str) -> list[ChunkWithPath]:
        """Chunks whose digest starts with ``prefix``, ordered by path then sequence.

        The prefix is the hex part of a content address (``blake3:`` already
        stripped). ``dig`` uses this to resolve either a full digest or the
        short prefix ``prospect`` prints; each chunk carries every path
        referencing its content.
        """
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN files f ON f.content_id = c.content_id "
                "WHERE c.digest LIKE ? ",
                (f"blake3:{prefix}%",),
            ).fetchall()
        mapped = [chunk_from_rows(group) for group in _group_rows(rows).values()]
        mapped.sort(key=lambda chunk: (chunk.primary.path, chunk.seq if chunk.seq is not None else 0))
        return mapped

    def get_chunk_neighbors(self, prefix: str, context: int) -> list[ChunkWithPath]:
        """Chunks adjacent to the addressed chunk within the same section.

        ``context`` is how many chunks to include on each side; the addressed
        chunk itself is excluded. Neighbors stay within the same content and
        heading chain, so a short prefix never pulls in unrelated sections;
        heading is NOT NULL, so an unheaded document simply shares ''.
        Returns an empty list when the prefix is unknown or ``context`` is
        non-positive.
        """
        if context <= 0:
            return []
        with self._lock:
            target = self._conn.execute(
                "SELECT id, content_id, seq, heading FROM chunks WHERE digest LIKE ? ORDER BY seq LIMIT 1",
                (f"blake3:{prefix}%",),
            ).fetchone()
            if target is None:
                return []
            target_id, content_id, seq, heading = target
            rows = self._conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN files f ON f.content_id = c.content_id "
                "WHERE c.content_id = ? AND c.id != ? AND c.seq BETWEEN ? AND ? "
                "AND c.heading = ? "
                "ORDER BY c.seq",
                (content_id, target_id, seq - context, seq + context, heading),
            ).fetchall()
        return [chunk_from_rows(group) for group in _group_rows(rows).values()]

    def list_files(self) -> list[FileRecord]:
        """All indexed paths with their content metadata, sorted by path."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT f.path, c.digest, f.mtime, f.size, f.status "
                "FROM files f JOIN contents c ON c.id = f.content_id "
                "ORDER BY f.path"
            ).fetchall()
        return [row_to_file(row) for row in rows]

    def get_file(self, path: str) -> FileRecord | None:
        """Metadata for one indexed path, or None when it is not indexed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT f.path, c.digest, f.mtime, f.size, f.status "
                "FROM files f JOIN contents c ON c.id = f.content_id "
                "WHERE f.path = ?",
                (path,),
            ).fetchone()
        return row_to_file(row) if row is not None else None

    def rebuild(self) -> None:
        """Drop everything (files included) and re-create the schema.

        The old database is snapshotted to ``<db_path>.bak`` first (via
        ``VACUUM INTO``) so a mistaken rebuild can be recovered manually.
        Requires an embedder to report the dimension; if none is available,
        nothing is changed.
        """
        with self._lock:
            embedder = self._require_embedder()
            try:
                dimension = embedder.dimension
                model_id = embedder.model_id
            except Exception as exc:
                raise EmbedderUnavailableError(
                    f"cannot rebuild index: could not determine embedding metadata: {exc}"
                ) from exc
            backup_path = Path(f"{self._db_path}.bak")
            backup_path.unlink(missing_ok=True)
            quoted = backup_path.as_posix().replace("'", "''")
            self._conn.execute(_VACUUM_INTO.format(quoted))
            with self._conn:
                drop_all(self._conn)
                create_schema(self._conn, dimension)
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                self._set_meta("model_id", model_id)
                self._set_meta("dimension", str(dimension))
            self._stored_model_id = model_id
            self._stored_dimension = dimension
            self._model_status = ModelStatus.MATCH
            self._model_checked = True

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

    def _ensure_content(self, digest: str) -> tuple[int, bool]:
        """Return ``(rowid, created)`` for the content with this digest."""
        row = self._conn.execute("SELECT id FROM contents WHERE digest = ?", (digest,)).fetchone()
        if row is not None:
            return int(row[0]), False
        cursor = self._conn.execute("INSERT INTO contents (digest) VALUES (?)", (digest,))
        return cast(int, cursor.lastrowid), True

    def _gc_content_if_orphaned(self, content_id: int) -> None:
        """Drop a content row once nothing references it.

        Orphanhood is derived by lookup rather than a stored refcount, so the
        invariant holds inside the surrounding write transaction without a
        counter that could drift.
        """
        referenced = self._conn.execute("SELECT 1 FROM files WHERE content_id = ? LIMIT 1", (content_id,)).fetchone()
        if referenced is not None:
            return
        chunk_rowids = [
            int(r[0]) for r in self._conn.execute("SELECT id FROM chunks WHERE content_id = ?", (content_id,))
        ]
        for rowid in chunk_rowids:
            self._conn.execute("DELETE FROM chunk_vectors WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM chunks WHERE content_id = ?", (content_id,))
        self._conn.execute("DELETE FROM contents WHERE id = ?", (content_id,))

    def _insert_chunks(self, content_id: int, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            cursor = self._conn.execute(
                "INSERT INTO chunks (digest, content_id, seq, text, heading, page) VALUES (?, ?, ?, ?, ?, ?)",
                (chunk.digest, content_id, chunk.seq, chunk.text, chunk.heading, chunk.page),
            )
            try:
                self._conn.execute(
                    "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                    (cast(int, cursor.lastrowid), json.dumps(vector)),
                )
            except sqlite3.OperationalError as exc:
                if "Dimension mismatch" in str(exc):
                    raise DimensionMismatchError(self.dimension, len(vector), operation="insert") from exc
                raise

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


def _group_rows(rows: list[tuple[Any, ...]]) -> dict[int, list[tuple[Any, ...]]]:
    """Group chunk-to-path join rows by chunk rowid, preserving query order."""
    grouped: dict[int, list[tuple[Any, ...]]] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)
    return grouped
