"""Schema definition for the index database.

Owns the DDL: tables, the vec0 virtual table, the FTS5 external-content
table, and the sync triggers. ``SCHEMA_VERSION`` guards compatibility — a
database written by an incompatible version is refused at open time and
requires an explicit rebuild.
"""

from __future__ import annotations

import sqlite3

# Bump when the schema changes incompatibly; a mismatch makes the store
# refuse to open until an explicit rebuild.
SCHEMA_VERSION = 1

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


def create_schema(conn: sqlite3.Connection, dimension: int) -> None:
    """Create every table, virtual table, and trigger on a fresh database."""
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    conn.execute(
        """
        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
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
    conn.execute(
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
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            embedding FLOAT[{dimension}]
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            chunk_id UNINDEXED,
            content='chunks',
            content_rowid='id'
        )
        """
    )
    conn.execute(_TRIGGER_CHUNKS_AI)
    conn.execute(_TRIGGER_CHUNKS_AD)
    conn.execute(_TRIGGER_CHUNKS_AU)


def drop_all(conn: sqlite3.Connection) -> None:
    """Drop every table of the index schema."""
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute("DROP TABLE IF EXISTS chunk_vectors")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS files")
    conn.execute("DROP TABLE IF EXISTS meta")
