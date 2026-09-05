//! Schema definition for the index database.
//!
//! Owns the DDL: tables, the vec0 virtual table, the FTS5 external-content
//! table, and the sync triggers. [`SCHEMA_VERSION`] guards compatibility — a
//! database written by an incompatible version is refused at open time and
//! requires an explicit rebuild.
//!
//! The schema separates *content* from *references*: `contents` holds each
//! unique indexed document once, keyed by its `blake3:<hex>` digest (an
//! inode), while `files` maps workspace paths onto that content. Identical
//! files at several paths share one set of chunks/vectors/FTS rows; a content
//! row lives until its last referencing path disappears.

use rusqlite::Connection;

/// Bump when the schema changes incompatibly; a mismatch makes the store
/// refuse to open until an explicit rebuild.
pub const SCHEMA_VERSION: u32 = 1;

/// FTS5 tokenizers the schema accepts. `tokenize_clause` is interpolated
/// into DDL, so it must stay on this whitelist (mirrors the Python
/// `FtsConfig.strategy` Literal).
const TOKENIZERS: &[&str] = &["unicode61", "trigram", "simple", "jieba"];

/// Native tokenizers shipped as a shared library; not implemented in the
/// Rust rewrite yet (Phase 3), so creating a schema with one is an error
/// with a clear hint instead of SQLite's bare "no such tokenizer".
const NATIVE_TOKENIZERS: &[&str] = &["simple", "jieba"];

/// Create every table, virtual table, index, and trigger on a fresh database.
///
/// `dimension` sizes the vec0 `embedding` column; `tokenize_clause` selects
/// the FTS5 tokenizer. Both are recorded in `meta` by the caller and
/// validated on later opens.
pub fn create_schema(
    conn: &Connection,
    dimension: u32,
    tokenize_clause: &str,
) -> crate::Result<()> {
    if dimension == 0 {
        return Err(crate::Error::Store(format!(
            "dimension must be positive, got {dimension}"
        )));
    }
    if !TOKENIZERS.contains(&tokenize_clause) {
        return Err(crate::Error::Store(format!(
            "unknown tokenizer {tokenize_clause:?}; choose from {}",
            TOKENIZERS.join(", ")
        )));
    }
    if NATIVE_TOKENIZERS.contains(&tokenize_clause) {
        return Err(crate::Error::Store(format!(
            "tokenizer {tokenize_clause:?} needs the native extension, which \
             the Rust rewrite does not implement yet; set fts.strategy = \"unicode61\""
        )));
    }

    conn.execute_batch(
        "CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE contents (
            id     INTEGER PRIMARY KEY,
            digest TEXT NOT NULL UNIQUE
        );
        CREATE TABLE files (
            id         INTEGER PRIMARY KEY,
            path       TEXT UNIQUE NOT NULL,
            content_id INTEGER NOT NULL REFERENCES contents(id),
            mtime      REAL NOT NULL,
            size       INTEGER NOT NULL,
            status     TEXT NOT NULL DEFAULT 'fresh'
                       CHECK (status IN ('fresh', 'stale'))
        );
        CREATE TABLE chunks (
            id         INTEGER PRIMARY KEY,
            digest     TEXT NOT NULL,
            content_id INTEGER NOT NULL REFERENCES contents(id) ON DELETE CASCADE,
            seq        INTEGER NOT NULL,
            text       TEXT NOT NULL,
            heading    TEXT NOT NULL DEFAULT '',
            page       INTEGER,
            UNIQUE (content_id, seq)
        );
        CREATE INDEX idx_chunks_digest ON chunks(digest);",
    )?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            embedding FLOAT[{dimension}]
        );"
    ))?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            tokenize='{tokenize_clause}',
            content='chunks',
            content_rowid='id'
        );"
    ))?;

    conn.execute_batch(
        "CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        END;
        CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;",
    )?;

    Ok(())
}
