//! SQLite-backed store: lifecycle, metadata, and query primitives.
//!
//! 1a scope: open an existing database, read metadata, list files, mark
//! stale. 1b adds the creation path: `Store::open` builds the full schema
//! (including vec0 and FTS5) on a fresh database, given the vector
//! dimension and tokenizer.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Once;

use rusqlite::Connection;

use crate::index::records::{FileRecord, FileStatus};
use crate::index::schema;
use crate::relpath::WorkspacePath;

/// Schema version; must match the database to open it.
pub use crate::index::schema::SCHEMA_VERSION;

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
/// [`Store::open`] creates a fresh database (full schema) when the file does
/// not exist yet, or opens an existing one after validating its metadata.
/// [`Store::open_existing`] is the read-only path used by `survey`.
#[derive(Debug)]
pub struct Store {
    #[allow(dead_code)]
    path: PathBuf,
    conn: Connection,
    meta: IndexMeta,
}

impl Store {
    /// Open an index database, creating it when it does not exist.
    ///
    /// Creation needs the vector dimension (sizes the vec0 table) and the
    /// FTS5 tokenizer; both are recorded in `meta` and validated on later
    /// opens. Fails if the database exists but is incompatible.
    pub fn open(path: &Path, dimension: u32, tokenizer: &str) -> crate::Result<Self> {
        if path.is_file() {
            Self::open_existing(path)
        } else {
            Self::create(path, dimension, tokenizer)
        }
    }

    /// Create a fresh index database with the full schema.
    fn create(path: &Path, dimension: u32, tokenizer: &str) -> crate::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // vec0 is a sqlite-vec virtual table; register the extension as an
        // auto-extension so every connection (this one and later opens) can
        // use it. Registration is process-global, so guard it with `Once`.
        register_vec_extension();

        let conn = Connection::open(path)?;
        configure_connection(&conn)?;
        schema::create_schema(&conn, dimension, tokenizer)?;
        write_meta(&conn, dimension, tokenizer)?;
        let meta = read_meta(&conn)?;

        Ok(Self {
            path: path.to_path_buf(),
            conn,
            meta,
        })
    }

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

/// Register the sqlite-vec extension as a SQLite auto-extension.
///
/// `sqlite3_auto_extension` is process-global and affects every connection
/// opened afterwards, so it runs exactly once via [`Once`]. The `transmute`
/// mirrors sqlite-vec's own test: the entry point is a plain `extern "C"`
/// function, but rusqlite's binding types it with the extension API
/// signature.
fn register_vec_extension() {
    static REGISTER_VEC: Once = Once::new();
    REGISTER_VEC.call_once(|| {
        // The transmute target type is fixed by rusqlite's binding, not by
        // this call site, so clippy cannot infer it.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
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

/// Write the metadata header on a freshly created database.
///
/// `model_id` is empty until the embedding layer lands (1c); dimension and
/// tokenizer are the values the schema was built with.
fn write_meta(conn: &Connection, dimension: u32, tokenizer: &str) -> crate::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params!["schema_version", SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params!["model_id", ""],
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

#[cfg(test)]
mod tests {
    use super::*;

    /// All tables (real and virtual) the schema must create.
    const EXPECTED_TABLES: &[&str] = &[
        "chunks",
        "chunks_fts",
        "chunk_vectors",
        "contents",
        "files",
        "meta",
    ];

    fn table_names(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type IN ('table', 'view')
                 ORDER BY name",
            )
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    #[test]
    fn open_creates_full_schema() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join(".lode").join("index.db");

        let store = Store::open(&db, 512, "unicode61").unwrap();

        // Metadata header is recorded.
        assert_eq!(store.meta().schema_version, SCHEMA_VERSION.to_string());
        assert_eq!(store.meta().dimension, 512);
        assert_eq!(store.meta().tokenizer, "unicode61");

        // Every table and virtual table exists.
        let tables = table_names(&store.conn);
        for expected in EXPECTED_TABLES {
            assert!(
                tables.iter().any(|t| t == expected),
                "missing table {expected}"
            );
        }
    }

    #[test]
    fn open_reopens_existing_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        Store::open(&db, 256, "unicode61").unwrap();

        let reopened = Store::open_existing(&db).unwrap();
        assert_eq!(reopened.meta().dimension, 256);
        assert_eq!(reopened.meta().tokenizer, "unicode61");
    }

    #[test]
    fn open_rejects_zero_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let err = Store::open(&db, 0, "simple").unwrap_err();
        assert!(err.to_string().contains("dimension must be positive"));
    }

    #[test]
    fn open_rejects_unknown_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let err = Store::open(&db, 512, "bogus").unwrap_err();
        assert!(err.to_string().contains("unknown tokenizer"));
    }

    #[test]
    fn open_rejects_native_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let err = Store::open(&db, 512, "simple").unwrap_err();
        assert!(err.to_string().contains("native extension"));
    }

    #[test]
    fn open_existing_missing_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("nope.db");

        let err = Store::open_existing(&db).unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }
}
