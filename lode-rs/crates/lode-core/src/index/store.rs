//! SQLite-backed store: lifecycle, metadata, and query primitives.
//!
//! 1a scope: open an existing database, read metadata, list files, mark
//! stale. Schema creation (which requires the embedder's dimension) is
//! deferred to Phase 1b.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use rusqlite::Connection;

use crate::index::records::{FileRecord, FileStatus};
use crate::relpath::WorkspacePath;

/// Schema version; must match the database to open it.
pub const SCHEMA_VERSION: u32 = 1;

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

/// SQLite index store.
///
/// Opens an existing database (never creates one). The schema creation path
/// — which needs the embedder's vector dimension — is deferred to Phase 1b.
pub struct Store {
    #[allow(dead_code)]
    path: PathBuf,
    conn: Connection,
    meta: IndexMeta,
}

impl Store {
    /// Open an existing index database.
    ///
    /// Reads the `meta` header, validates schema version and metadata, and
    /// returns a ready-to-use store. Fails if:
    /// - the database does not exist
    /// - the schema version is incompatible
    /// - required metadata keys are missing
    pub fn open_existing(path: &Path) -> crate::Result<Self> {
        if !path.is_file() {
            return Err(crate::Error::Store(format!(
                "database does not exist: {}",
                path.display()
            )));
        }

        let conn = Connection::open(path)?;
        configure_connection(&conn)?;
        let meta = read_meta(&conn)?;

        Ok(Self {
            path: path.to_path_buf(),
            conn,
            meta,
        })
    }

    /// The stored index metadata.
    pub fn meta(&self) -> &IndexMeta {
        &self.meta
    }

    /// All indexed paths with their content metadata, sorted by path.
    pub fn list_files(&self) -> crate::Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, c.digest, f.mtime, f.size, f.status
             FROM files f JOIN contents c ON c.id = f.content_id
             ORDER BY f.path",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(FileRecord {
                path: WorkspacePath::from_posix(row.get::<_, String>(0)?),
                digest: row.get(1)?,
                mtime: row.get(2)?,
                size: row.get(3)?,
                status: FileStatus::from_str(&row.get::<_, String>(4)?)
                    .unwrap_or(FileStatus::Stale),
            })
        })?;

        let mut files = Vec::new();
        for row in rows {
            files.push(row?);
        }
        Ok(files)
    }

    /// Mark a path's snapshot as outdated (content changed, sync pending).
    ///
    /// Unknown paths are ignored: a new file that fails before its first
    /// successful indexing leaves no row behind.
    pub fn mark_stale(&self, path: &WorkspacePath) -> crate::Result<()> {
        self.conn.execute(
            "UPDATE files SET status = ?1 WHERE path = ?2",
            rusqlite::params![FileStatus::Stale.as_str(), path.as_str()],
        )?;
        Ok(())
    }

    /// Metadata for one indexed path, or `None` when it is not indexed.
    pub fn get_file(&self, path: &WorkspacePath) -> crate::Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, c.digest, f.mtime, f.size, f.status
             FROM files f JOIN contents c ON c.id = f.content_id
             WHERE f.path = ?1",
        )?;

        let mut rows = stmt.query_map(rusqlite::params![path.as_str()], |row| {
            Ok(FileRecord {
                path: WorkspacePath::from_posix(row.get::<_, String>(0)?),
                digest: row.get(1)?,
                mtime: row.get(2)?,
                size: row.get(3)?,
                status: FileStatus::from_str(&row.get::<_, String>(4)?)
                    .unwrap_or(FileStatus::Stale),
            })
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Close the database connection.
    pub fn close(self) {
        drop(self);
    }
}

/// Configure connection pragmas (WAL, busy timeout, foreign keys).
fn configure_connection(conn: &Connection) -> crate::Result<()> {
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

/// Read and validate the `meta` table.
fn read_meta(conn: &Connection) -> crate::Result<IndexMeta> {
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
fn meta_get(conn: &Connection, key: &str) -> crate::Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
    let mut rows = stmt.query_map(rusqlite::params![key], |row| row.get::<_, String>(0))?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}
