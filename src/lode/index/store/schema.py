"""Schema definition for the index database.

Owns the DDL: tables, the vec0 virtual table, the FTS5 external-content
table, and the sync triggers. ``SCHEMA_VERSION`` guards compatibility — a
database written by an incompatible version is refused at open time and
requires an explicit rebuild.

The schema separates *content* from *references*: ``contents`` holds each
unique indexed document once, keyed by its ``blake3:<hex>`` digest (an
inode), while ``files`` maps workspace paths onto that content. Identical
files at several paths share one set of chunks/vectors/FTS rows; a content
row lives until its last referencing path disappears.
"""

from __future__ import annotations

import sqlite3

# Bump when the schema changes incompatibly; a mismatch makes the store
# refuse to open until an explicit rebuild.
SCHEMA_VERSION = 1

_TRIGGER_CHUNKS_AI = """
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END
"""

_TRIGGER_CHUNKS_AD = """
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END
"""

_TRIGGER_CHUNKS_AU = """
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END
"""


def create_schema(conn: sqlite3.Connection, dimension: int, tokenize_clause: str = "unicode61") -> None:
    """Create every table, virtual table, index, and trigger on a fresh database."""
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
        CREATE TABLE contents (
            id     INTEGER PRIMARY KEY,
            digest TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE files (
            id         INTEGER PRIMARY KEY,
            path       TEXT UNIQUE NOT NULL,
            content_id INTEGER NOT NULL REFERENCES contents(id),
            mtime      REAL NOT NULL,
            size       INTEGER NOT NULL,
            status     TEXT NOT NULL DEFAULT 'fresh'
                       CHECK (status IN ('fresh', 'stale'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks (
            id         INTEGER PRIMARY KEY,
            digest     TEXT NOT NULL,
            content_id INTEGER NOT NULL REFERENCES contents(id) ON DELETE CASCADE,
            seq        INTEGER NOT NULL,
            text       TEXT NOT NULL,
            heading    TEXT NOT NULL DEFAULT '',
            page       INTEGER,
            UNIQUE (content_id, seq)
        )
        """
    )
    conn.execute("CREATE INDEX idx_chunks_digest ON chunks(digest)")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            embedding FLOAT[{dimension}]
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            tokenize='{tokenize_clause}',
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
    conn.execute("DROP TABLE IF EXISTS contents")
    conn.execute("DROP TABLE IF EXISTS meta")
