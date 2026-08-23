"""Tests for the SQLite index store.

Hermetic: an in-memory fake embedder, file-backed databases in tmp_path,
no network. The tests assert stored behavior (what lands in files/chunks/
vectors/FTS and how it is served back), not implementation details — the
raw-SQL probes open a *second* connection to read what the store wrote.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# The sqlite_vec package ships no type stubs; see test_sqlite_vec_smoke.py.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]

from lode.embeddings.base import Embedder
from lode.index import (
    DimensionMismatchError,
    EmbedderUnavailableError,
    FileRecord,
    FileStatus,
    ModelStatus,
    SchemaVersionError,
    Store,
)
from lode.ingestion import Chunk, chunk_id

DIM = 4


class FakeEmbedder(Embedder):
    """Deterministic in-memory embedder for hermetic store tests."""

    def __init__(
        self,
        *,
        model_id: str = "test-model",
        dimension: int = DIM,
        fail_model_id: bool = False,
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self.fail_model_id = fail_model_id
        self.dimension_calls = 0

    @property
    def model_id(self) -> str:
        if self.fail_model_id:
            raise RuntimeError("embedding endpoint is down")
        return self._model_id

    @property
    def dimension(self) -> int:
        self.dimension_calls += 1
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * self._dimension for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dimension


def make_chunks(texts: list[str]) -> tuple[list[Chunk], list[list[float]]]:
    chunks = [Chunk(id=chunk_id(text), text=text, seq=seq) for seq, text in enumerate(texts)]
    vectors = [[0.1 * (seq + 1), 0.2, 0.3, 0.4] for seq in range(len(texts))]
    return chunks, vectors


def file_record(path: str = "a.txt", *, digest: str = "blake3:aa", size: int = 1) -> FileRecord:
    return FileRecord(path=path, digest=digest, mtime=1.0, size=size)


def open_db(path: Path) -> sqlite3.Connection:
    """Second connection for reading what the store wrote (WAL-safe)."""
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "index.db"


# -- initialization ---------------------------------------------------------


def test_initialize_records_model_and_dimension(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder(model_id="m1", dimension=7)) as store:
        assert store.dimension == 7
        assert store.stored_model_id == "m1"
        assert store.model_status is ModelStatus.MATCH


def test_reopen_uses_stored_dimension_without_touching_embedder(db_path: Path) -> None:
    embedder = FakeEmbedder(model_id="m1", dimension=7)
    with Store(db_path, embedder) as store:
        assert store.dimension == 7
        assert embedder.dimension_calls == 1  # once, to create the schema

    # Reopening with a *different* reported dimension must not re-probe:
    # the dimension comes from meta, not the embedder (PLAN D7).
    reopened = FakeEmbedder(model_id="m1", dimension=99)
    with Store(db_path, reopened) as store:
        assert store.dimension == 7
        assert reopened.dimension_calls == 0
        assert store.model_status is ModelStatus.MATCH


def test_new_store_with_unreachable_embedder_fails_cleanly(db_path: Path) -> None:
    with pytest.raises(EmbedderUnavailableError):
        Store(db_path, FakeEmbedder(fail_model_id=True))


# -- schema version ---------------------------------------------------------


def test_schema_mismatch_refuses_to_open(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder()):
        pass

    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError):
        Store(db_path, FakeEmbedder())


# -- model detection --------------------------------------------------------


def test_model_mismatch_is_reported_not_fatal(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder(model_id="m1")) as store:
        assert store.model_status is ModelStatus.MATCH

    with Store(db_path, FakeEmbedder(model_id="m2")) as store:
        assert store.model_status is ModelStatus.MISMATCH


def test_unreachable_embedder_keeps_store_open(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder(model_id="m1")) as store:
        pass

    with Store(db_path, FakeEmbedder(fail_model_id=True)) as store:
        assert store.model_status is ModelStatus.UNKNOWN
        assert store.dimension == DIM  # still fully usable


# -- replace_file -----------------------------------------------------------


def test_replace_file_writes_files_chunks_vectors_and_fts(db_path: Path) -> None:
    chunks, vectors = make_chunks(["hello world", "second chunk"])
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record(), chunks, vectors)
        assert store.get_file("a.txt") == file_record()
        assert len(store.list_files()) == 1

    conn = open_db(db_path)
    assert count(conn, "SELECT count(*) FROM files") == 1
    assert count(conn, "SELECT count(*) FROM chunks") == 2
    # FTS is kept in sync by triggers (external content table).
    hits = conn.execute("SELECT text FROM chunks_fts WHERE chunks_fts MATCH 'hello'").fetchall()
    assert [row[0] for row in hits] == ["hello world"]
    # Dense vectors are queryable via KNN.
    rows = conn.execute(
        "SELECT rowid, distance FROM chunk_vectors "
        "WHERE embedding MATCH '[1.0, 0.0, 0.0, 0.0]' AND k = 10 ORDER BY distance"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][1] <= rows[1][1]
    conn.close()


def test_replace_file_with_no_chunks_keeps_file_record(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record(), [], [])
        assert store.get_file("a.txt") is not None


def test_replace_file_replaces_atomically(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record(), *make_chunks(["one", "two"]))
        store.replace_file(file_record(), *make_chunks(["only"]))

    conn = open_db(db_path)
    assert count(conn, "SELECT count(*) FROM files") == 1
    assert count(conn, "SELECT count(*) FROM chunks") == 1
    # No stale FTS rows from the previous version of the file.
    assert count(conn, "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'one'") == 0
    assert count(conn, "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'only'") == 1
    assert count(conn, "SELECT count(*) FROM chunk_vectors") == 1
    conn.close()


def test_replace_file_rolls_back_on_vector_error(db_path: Path) -> None:
    chunks, _ = make_chunks(["ok"])
    bad_vectors = [[0.1, 0.2]]  # 2 dimensions, table expects 4
    with Store(db_path, FakeEmbedder()) as store:
        with pytest.raises(DimensionMismatchError):
            store.replace_file(file_record(), chunks, bad_vectors)
        # The failed transaction left no partial state behind.
        assert store.get_file("a.txt") is None
        assert store.list_files() == []


def test_replace_file_rejects_mismatched_chunk_vector_counts(db_path: Path) -> None:
    chunks, vectors = make_chunks(["a", "b"])
    with Store(db_path, FakeEmbedder()) as store, pytest.raises(ValueError):
        store.replace_file(file_record(), chunks, vectors[:1])


# -- remove_file ------------------------------------------------------------


def test_remove_file_cleans_files_chunks_vectors_and_fts(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record(), *make_chunks(["hello world"]))
        store.remove_file("a.txt")
        assert store.get_file("a.txt") is None
        assert store.list_files() == []

    conn = open_db(db_path)
    assert count(conn, "SELECT count(*) FROM files") == 0
    assert count(conn, "SELECT count(*) FROM chunks") == 0
    assert count(conn, "SELECT count(*) FROM chunk_vectors") == 0
    assert count(conn, "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'hello'") == 0
    conn.close()


def test_remove_missing_file_is_a_noop(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder()) as store:
        store.remove_file("nope.txt")
        assert store.list_files() == []


# -- find_chunks_by_digest ---------------------------------------------------


def test_find_chunks_by_digest_prefix_matches(db_path: Path) -> None:
    chunks = [
        Chunk(id="blake3:aaaa1111bbbb", text="first", seq=0),
        Chunk(id="blake3:aaaa2222cccc", text="second", seq=1),
        Chunk(id="blake3:bbbbeeeeffff", text="third", seq=2),
    ]
    vectors = [[0.1, 0.2, 0.3, 0.4] for _ in chunks]
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record("a.txt", digest="blake3:aa", size=1), chunks, vectors)

        # A shared prefix resolves to both, ordered by sequence.
        assert [c.text for c in store.find_chunks_by_digest("aaaa")] == ["first", "second"]
        assert [c.text for c in store.find_chunks_by_digest("bbbb")] == ["third"]
        # A full digest resolves to exactly one.
        assert [c.text for c in store.find_chunks_by_digest("aaaa1111bbbb")] == ["first"]
        # No match -> empty list.
        assert store.find_chunks_by_digest("ffff") == []


def test_find_chunks_by_digest_returns_provenance(db_path: Path) -> None:
    chunks = [Chunk(id="blake3:cccc1111", text="content", seq=0, heading="Section 1", page=2)]
    vectors = [[0.1, 0.2, 0.3, 0.4]]
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record("report.pdf", digest="blake3:cc", size=3), chunks, vectors)
        found = store.find_chunks_by_digest("cccc")

    assert len(found) == 1
    chunk = found[0]
    assert chunk.chunk_id == "blake3:cccc1111"
    assert chunk.text == "content"
    assert chunk.path == "report.pdf"
    assert chunk.heading == "Section 1"
    assert chunk.page == 2
    assert chunk.file_status is FileStatus.CURRENT


# -- rebuild ----------------------------------------------------------------


def test_rebuild_snapshots_then_resets(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder(model_id="m1")) as store:
        store.replace_file(file_record(), *make_chunks(["hello world"]))

    backup = Path(str(db_path) + ".bak")
    with Store(db_path, FakeEmbedder(model_id="m2", dimension=8)) as store:
        assert store.model_status is ModelStatus.MISMATCH
        store.rebuild()
        assert store.model_status is ModelStatus.MATCH
        assert store.dimension == 8
        assert store.stored_model_id == "m2"
        assert store.list_files() == []

    # The old database survived the rebuild as a snapshot.
    assert backup.exists()
    conn = sqlite3.connect(str(backup))
    assert count(conn, "SELECT count(*) FROM files") == 1
    conn.close()


def test_rebuild_with_unreachable_embedder_changes_nothing(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder(model_id="m1")) as store:
        store.replace_file(file_record(), *make_chunks(["hello world"]))

    with Store(db_path, FakeEmbedder(fail_model_id=True)) as store:
        with pytest.raises(EmbedderUnavailableError):
            store.rebuild()
        # Nothing was dropped and no backup was written.
        assert store.list_files() == [file_record()]
        assert not Path(str(db_path) + ".bak").exists()


# -- pragmas ----------------------------------------------------------------


def test_wal_mode_is_enabled(db_path: Path) -> None:
    with Store(db_path, FakeEmbedder()) as store:
        store.replace_file(file_record(), *make_chunks(["hello world"]))

    conn = open_db(db_path)
    mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    conn.close()
    assert mode == "wal"
