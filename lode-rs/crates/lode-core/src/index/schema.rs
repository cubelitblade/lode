#![warn(clippy::pedantic)]

//! Schema definition for the index database.
//!
//! Owns the DDL: tables, the vec0 virtual table, the FTS5 external-content
//! table, and the sync triggers. [`SCHEMA_VERSION`] guards compatibility — a
//! database written by an incompatible version is refused at open time and
//! requires an explicit rebuild.
//!
//! The schema separates *content* from *references*: `contents` holds each
//! unique indexed document once, keyed by its `blake3:<hex>` digest and
//! extractor family, while `files` maps workspace paths onto that content.
//! Identical files handled by the same extractor family share one set of
//! chunks/vectors/FTS rows; a content row lives until its last referencing
//! path disappears.

use rusqlite::Connection;

/// Bump when the schema changes incompatibly; a mismatch makes the store
/// refuse to open until an explicit rebuild.
/// Version 5 adds optional character spans for overlap-aware context output.
/// Older indexes must be rebuilt explicitly rather than silently reused.
pub const SCHEMA_VERSION: u32 = 5;

/// FTS5 tokenizers the schema accepts. `tokenize_clause` is interpolated
/// into DDL, so it must stay on this whitelist (mirrors the Python
/// `FtsConfig.strategy` Literal).
const TOKENIZERS: &[&str] = &["unicode61", "trigram", "simple", "jieba"];

/// Resolve a configured strategy to the tokenizer name stored in the FTS5
/// DDL. `jieba` is a query strategy layered over the same simple document
/// tokenizer, so switching between the two does not rebuild the index.
#[must_use]
pub fn physical_tokenizer(strategy: &str) -> Option<&'static str> {
    match strategy {
        "simple" | "jieba" => Some("simple"),
        "unicode61" => Some("unicode61"),
        "trigram" => Some("trigram"),
        _ => None,
    }
}

/// Create every table, virtual table, index, and trigger on a fresh database.
///
/// `dimension` sizes the vec0 `embedding` column; `tokenize_clause` selects
/// the FTS5 tokenizer. Both are recorded in `meta` by the caller and
/// validated on later opens.
///
/// # Errors
///
/// Fails when `dimension` is zero, the tokenizer is not on the whitelist, or
/// SQLite rejects the DDL.
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
    let Some(physical_tokenizer) = physical_tokenizer(tokenize_clause) else {
        return Err(crate::Error::Store(format!(
            "unknown tokenizer {tokenize_clause:?}; choose from {}",
            TOKENIZERS.join(", ")
        )));
    };

    conn.execute_batch(
        "CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE contents (
            id     INTEGER PRIMARY KEY,
            digest TEXT NOT NULL,
            extractor TEXT NOT NULL,
            UNIQUE (digest, extractor)
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
        CREATE TABLE chunk_spans (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            segment  INTEGER NOT NULL,
            start    INTEGER NOT NULL,
            end      INTEGER NOT NULL,
            CHECK (segment >= 0),
            CHECK (start >= 0),
            CHECK (end >= start)
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
            tokenize='{physical_tokenizer}',
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
