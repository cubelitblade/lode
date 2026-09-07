#![warn(clippy::pedantic)]

//! Index metadata: the `meta` table header written at creation and read
//! and validated at open time.

use rusqlite::Connection;

/// Schema version; must match the database to open it.
use crate::index::schema::SCHEMA_VERSION;

/// Busy timeout in milliseconds (matches Python).
const BUSY_TIMEOUT_MS: i32 = 5000;

/// The index metadata header, read from the `meta` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    pub schema_version: String,
    pub model_id: String,
    pub dimension: u32,
    pub tokenizer: String,
}

pub(super) fn read_meta(conn: &Connection) -> crate::Result<IndexMeta> {
    let version = meta_get(conn, "schema_version")?.unwrap_or_default();
    if version != SCHEMA_VERSION.to_string() {
        return Err(crate::Error::Store(format!(
            "database schema version {version:?} is incompatible with \
             supported version {SCHEMA_VERSION}; run an explicit rebuild"
        )));
    }

    let model_id = meta_get(conn, "model_id")?.unwrap_or_default();
    let dimension_str = meta_get(conn, "dimension")?.unwrap_or_default();
    let tokenizer = meta_get(conn, "tokenizer")?.unwrap_or_else(|| "unicode61".to_string());

    let dimension: u32 = dimension_str.parse().map_err(|_| {
        crate::Error::Store(format!(
            "invalid dimension metadata: {dimension_str:?}; run an explicit rebuild"
        ))
    })?;

    Ok(IndexMeta {
        schema_version: version,
        model_id,
        dimension,
        tokenizer,
    })
}

/// Read a single key from the `meta` table.
pub(super) fn meta_get(conn: &Connection, key: &str) -> crate::Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
    let mut rows = stmt.query_map(rusqlite::params![key], |row| row.get::<_, String>(0))?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

/// Write the metadata header on a freshly created database.
///
/// `model_id`, `dimension`, and `tokenizer` are the values the schema was
/// built with; `model_id` comes from the embedder (mirroring Python's
/// `_initialize`, which writes `embedder.model_id`).
pub(super) fn write_meta(
    conn: &Connection,
    model_id: &str,
    dimension: u32,
    tokenizer: &str,
) -> crate::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params!["schema_version", SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params!["model_id", model_id],
    )?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params!["dimension", dimension.to_string()],
    )?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params!["tokenizer", tokenizer],
    )?;
    Ok(())
}
pub(super) fn configure_connection(conn: &Connection) -> crate::Result<()> {
    // `journal_mode=WAL` and `busy_timeout` return a result row when set, so
    // they cannot go through `execute_batch` (which rejects statements that
    // return results). `foreign_keys=ON` and `case_sensitive_like=ON` return
    // no rows, so they must not go through `query_row`.
    conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    conn.query_row(
        &format!("PRAGMA busy_timeout={BUSY_TIMEOUT_MS}"),
        [],
        |_| Ok(()),
    )?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;
    conn.execute_batch("PRAGMA case_sensitive_like=ON;")?;
    Ok(())
}
