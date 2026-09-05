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
use rusqlite::OptionalExtension;

use crate::index::records::{FileRecord, FileStatus};
use crate::index::schema;
use crate::ingestion::types::Chunk;
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
    pub fn mark_stale(&mut self, path: &WorkspacePath) -> crate::Result<()> {
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

    /// Point `path` at already-indexed content; report whether it existed.
    ///
    /// Unlike [`replace_file`], no chunks are written — the caller claims
    /// the content addressed by `record.digest` is already indexed. When it
    /// is not, nothing changes and `false` is returned.
    pub fn reference_file(&mut self, record: &FileRecord) -> crate::Result<bool> {
        let tx = self.conn.transaction()?;

        let content_id: Option<i64> = tx
            .query_row(
                "SELECT id FROM contents WHERE digest = ?1",
                rusqlite::params![record.digest],
                |row| row.get(0),
            )
            .optional()?;

        let Some(content_id) = content_id else {
            // Content not indexed yet — nothing to reference.
            tx.commit()?;
            return Ok(false);
        };

        // Get previous content_id for GC.
        let previous: Option<i64> = tx
            .query_row(
                "SELECT content_id FROM files WHERE path = ?1",
                rusqlite::params![record.path.as_str()],
                |row| row.get(0),
            )
            .optional()?;

        tx.execute(
            "INSERT INTO files (path, content_id, mtime, size, status)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(path) DO UPDATE SET
                content_id = excluded.content_id,
                mtime = excluded.mtime,
                size = excluded.size,
                status = excluded.status",
            rusqlite::params![
                record.path.as_str(),
                content_id,
                record.mtime,
                record.size as i64,
                record.status.as_str(),
            ],
        )?;

        if let Some(old_id) = previous {
            if old_id != content_id {
                gc_content_if_orphaned(&tx, old_id)?;
            }
        }

        tx.commit()?;
        Ok(true)
    }

    /// Atomically replace `record.path`'s content, writing chunks and vectors.
    ///
    /// Creates the content row when this is its first reference; reuses
    /// existing content when another path already indexed the same digest.
    /// When the path moves away from a previous content, that content is
    /// dropped once its last reference disappears.
    ///
    /// `vectors` is `None` for text-only writes (no embedding layer wired
    /// yet); when present it must be one per chunk and match the index
    /// dimension (checked only when the content is actually written,
    /// mirroring Python).
    ///
    /// Returns whether the content was newly created.
    pub fn replace_file(
        &mut self,
        record: &FileRecord,
        chunks: &[Chunk],
        vectors: Option<&[Vec<f32>]>,
    ) -> crate::Result<bool> {
        if let Some(vectors) = vectors {
            if chunks.len() != vectors.len() {
                return Err(crate::Error::Store(format!(
                    "got {} chunks but {} vectors",
                    chunks.len(),
                    vectors.len()
                )));
            }
        }

        let tx = self.conn.transaction()?;

        let (content_id, created) = ensure_content(&tx, &record.digest)?;

        let previous: Option<i64> = tx
            .query_row(
                "SELECT content_id FROM files WHERE path = ?1",
                rusqlite::params![record.path.as_str()],
                |row| row.get(0),
            )
            .optional()?;

        tx.execute(
            "INSERT INTO files (path, content_id, mtime, size, status)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(path) DO UPDATE SET
                content_id = excluded.content_id,
                mtime = excluded.mtime,
                size = excluded.size,
                status = excluded.status",
            rusqlite::params![
                record.path.as_str(),
                content_id,
                record.mtime,
                record.size as i64,
                record.status.as_str(),
            ],
        )?;

        // Write chunks and vectors only when content is newly created.
        if created {
            insert_chunks(&tx, content_id, chunks, vectors, self.meta.dimension)?;
        }

        if let Some(old_id) = previous {
            if old_id != content_id {
                gc_content_if_orphaned(&tx, old_id)?;
            }
        }

        tx.commit()?;
        Ok(created)
    }

    /// Delete a path reference; drop its content when this was the last one.
    pub fn remove_file(&mut self, path: &WorkspacePath) -> crate::Result<()> {
        let tx = self.conn.transaction()?;

        let content_id: Option<i64> = tx
            .query_row(
                "SELECT content_id FROM files WHERE path = ?1",
                rusqlite::params![path.as_str()],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(cid) = content_id {
            tx.execute(
                "DELETE FROM files WHERE path = ?1",
                rusqlite::params![path.as_str()],
            )?;
            gc_content_if_orphaned(&tx, cid)?;
        }

        tx.commit()?;
        Ok(())
    }
}

/// Return `(content_id, created)` for the content with this digest.
fn ensure_content(conn: &Connection, digest: &str) -> crate::Result<(i64, bool)> {
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM contents WHERE digest = ?1",
            rusqlite::params![digest],
            |row| row.get(0),
        )
        .optional()?;

    match existing {
        Some(id) => Ok((id, false)),
        None => {
            conn.execute(
                "INSERT INTO contents (digest) VALUES (?1)",
                rusqlite::params![digest],
            )?;
            Ok((conn.last_insert_rowid(), true))
        }
    }
}

/// Drop a content row once nothing references it.
///
/// Orphanhood is derived by lookup rather than a stored refcount, so the
/// invariant holds inside the surrounding write transaction.
fn gc_content_if_orphaned(conn: &Connection, content_id: i64) -> crate::Result<()> {
    let referenced: bool = conn
        .query_row(
            "SELECT 1 FROM files WHERE content_id = ?1 LIMIT 1",
            rusqlite::params![content_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if referenced {
        return Ok(());
    }

    // Collect chunk rowids before cascade delete removes them.
    let rowids: Vec<i64> = {
        let mut stmt = conn.prepare("SELECT id FROM chunks WHERE content_id = ?1")?;
        let rows = stmt.query_map(rusqlite::params![content_id], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };

    for rowid in &rowids {
        conn.execute(
            "DELETE FROM chunk_vectors WHERE rowid = ?1",
            rusqlite::params![rowid],
        )?;
    }

    // Cascade: DELETE contents → chunks (triggers handle FTS5 cleanup).
    conn.execute(
        "DELETE FROM contents WHERE id = ?1",
        rusqlite::params![content_id],
    )?;
    Ok(())
}

/// Write chunk rows (FTS5 sync triggers fire automatically) and, when
/// vectors are present, their embeddings into the vec0 table.
///
/// Vectors are serialized as JSON arrays (what sqlite-vec expects) and keyed
/// by the chunk's rowid. A vector whose width differs from the index
/// dimension is refused with [`crate::Error::DimensionMismatch`], mirroring
/// Python's `DimensionMismatchError`.
fn insert_chunks(
    conn: &Connection,
    content_id: i64,
    chunks: &[Chunk],
    vectors: Option<&[Vec<f32>]>,
    dimension: u32,
) -> crate::Result<()> {
    let mut stmt = conn.prepare(
        "INSERT INTO chunks (digest, content_id, seq, text, heading, page)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    let mut vec_stmt =
        conn.prepare("INSERT INTO chunk_vectors (rowid, embedding) VALUES (?1, ?2)")?;

    for (i, chunk) in chunks.iter().enumerate() {
        stmt.execute(rusqlite::params![
            chunk.digest,
            content_id,
            chunk.seq as i64,
            chunk.text,
            chunk.heading,
            chunk.page.map(|p| p as i64),
        ])?;
        if let Some(vectors) = vectors {
            let vector = &vectors[i];
            if vector.len() != dimension as usize {
                return Err(crate::Error::DimensionMismatch {
                    stored: dimension,
                    current: vector.len() as u32,
                });
            }
            let rowid = conn.last_insert_rowid();
            let embedding = serde_json::to_string(vector)
                .map_err(|e| crate::Error::Store(format!("could not serialize embedding: {e}")))?;
            vec_stmt.execute(rusqlite::params![rowid, embedding])?;
        }
    }

    Ok(())
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

    // -- reference_file / replace_file / remove_file tests --

    use crate::ingestion::types::Chunk;

    fn make_record(path: &str, digest: &str, mtime: f64, size: u64) -> FileRecord {
        FileRecord {
            path: WorkspacePath::from_posix(path),
            digest: digest.to_string(),
            mtime,
            size,
            status: FileStatus::Fresh,
        }
    }

    fn make_chunks(digest: &str, n: usize) -> Vec<Chunk> {
        (0..n)
            .map(|i| Chunk {
                digest: digest.to_string(),
                text: format!("chunk {i}"),
                seq: i as u32,
                heading: String::new(),
                page: None,
            })
            .collect()
    }

    /// `n` vectors of `dimension` width, each entry equal to its index.
    fn make_vectors(n: usize, dimension: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| vec![i as f32; dimension]).collect()
    }

    #[test]
    fn reference_file_new_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let rec = make_record("a.txt", "blake3:aaa", 1.0, 100);
        assert!(!store.reference_file(&rec).unwrap());
        // Nothing was written — the content was not indexed yet.
        assert!(store.list_files().unwrap().is_empty());
    }

    #[test]
    fn reference_file_reuses_existing_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        // Create the content first.
        let r1 = make_record("a.txt", "blake3:same", 1.0, 100);
        store
            .replace_file(
                &r1,
                &make_chunks("blake3:same", 1),
                Some(&make_vectors(1, 128)),
            )
            .unwrap();

        // Second path references the existing content.
        let r2 = make_record("b.txt", "blake3:same", 2.0, 200);
        assert!(store.reference_file(&r2).unwrap());

        // Both paths point at the same content.
        let files = store.list_files().unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].digest, files[1].digest);
    }

    #[test]
    fn replace_file_creates_content_and_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let rec = make_record("doc.txt", "blake3:bbb", 1.0, 50);
        let chunks = make_chunks("blake3:bbb", 3);
        let created = store
            .replace_file(&rec, &chunks, Some(&make_vectors(3, 128)))
            .unwrap();
        assert!(created);

        // Chunks are in the database.
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 3);

        // Vectors are in the vec0 table, one per chunk.
        let vec_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunk_vectors", [], |row| row.get(0))
            .unwrap();
        assert_eq!(vec_count, 3);
    }

    #[test]
    fn replace_file_reuses_existing_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        // First file creates the content.
        let r1 = make_record("a.txt", "blake3:shared", 1.0, 100);
        let chunks = make_chunks("blake3:shared", 2);
        store
            .replace_file(&r1, &chunks, Some(&make_vectors(2, 128)))
            .unwrap();

        // Second file with same digest reuses content; no extra chunks.
        let r2 = make_record("b.txt", "blake3:shared", 2.0, 100);
        let created = store
            .replace_file(&r2, &chunks, Some(&make_vectors(2, 128)))
            .unwrap();
        assert!(!created);

        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2); // still 2, not 4

        let vec_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunk_vectors", [], |row| row.get(0))
            .unwrap();
        assert_eq!(vec_count, 2); // still 2, not 4
    }

    #[test]
    fn remove_file_drops_path_and_orphans_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let rec = make_record("only.txt", "blake3:ccc", 1.0, 10);
        let chunks = make_chunks("blake3:ccc", 1);
        store
            .replace_file(&rec, &chunks, Some(&make_vectors(1, 128)))
            .unwrap();

        store
            .remove_file(&WorkspacePath::from_posix("only.txt"))
            .unwrap();

        // Path gone, content orphaned and GC'd.
        assert!(store.list_files().unwrap().is_empty());
        let content_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM contents", [], |row| row.get(0))
            .unwrap();
        assert_eq!(content_count, 0);
        let vec_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunk_vectors", [], |row| row.get(0))
            .unwrap();
        assert_eq!(vec_count, 0);
    }

    #[test]
    fn remove_file_preserves_shared_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let chunks = make_chunks("blake3:ddd", 1);
        store
            .replace_file(
                &make_record("a.txt", "blake3:ddd", 1.0, 10),
                &chunks,
                Some(&make_vectors(1, 128)),
            )
            .unwrap();
        store
            .replace_file(
                &make_record("b.txt", "blake3:ddd", 2.0, 10),
                &chunks,
                Some(&make_vectors(1, 128)),
            )
            .unwrap();

        store
            .remove_file(&WorkspacePath::from_posix("a.txt"))
            .unwrap();

        // b.txt still references the content.
        assert_eq!(store.list_files().unwrap().len(), 1);
        let content_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM contents", [], |row| row.get(0))
            .unwrap();
        assert_eq!(content_count, 1);
    }

    #[test]
    fn replace_file_updates_existing_path() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let r1 = make_record("doc.txt", "blake3:v1", 1.0, 100);
        store
            .replace_file(
                &r1,
                &make_chunks("blake3:v1", 2),
                Some(&make_vectors(2, 128)),
            )
            .unwrap();

        // Same path, different content.
        let r2 = make_record("doc.txt", "blake3:v2", 2.0, 200);
        store
            .replace_file(
                &r2,
                &make_chunks("blake3:v2", 3),
                Some(&make_vectors(3, 128)),
            )
            .unwrap();

        let files = store.list_files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].digest, "blake3:v2");
        assert_eq!(files[0].mtime, 2.0);

        // Old content orphaned and GC'd.
        let content_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM contents", [], |row| row.get(0))
            .unwrap();
        assert_eq!(content_count, 1);
    }

    #[test]
    fn replace_file_rejects_chunk_vector_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let rec = make_record("doc.txt", "blake3:mmm", 1.0, 50);
        let chunks = make_chunks("blake3:mmm", 3);
        let err = store
            .replace_file(&rec, &chunks, Some(&make_vectors(2, 128)))
            .unwrap_err();
        assert!(err.to_string().contains("3 chunks but 2 vectors"));
    }

    #[test]
    fn replace_file_rejects_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let rec = make_record("doc.txt", "blake3:mmm", 1.0, 50);
        let chunks = make_chunks("blake3:mmm", 1);
        let err = store
            .replace_file(&rec, &chunks, Some(&make_vectors(1, 64)))
            .unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
        assert!(err.to_string().contains("stored 128"));
        assert!(err.to_string().contains("got 64"));

        // Nothing was written.
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn replace_file_writes_vectors_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();

        let rec = make_record("doc.txt", "blake3:vvv", 1.0, 50);
        let chunks = make_chunks("blake3:vvv", 2);
        let vectors = make_vectors(2, 128);
        store.replace_file(&rec, &chunks, Some(&vectors)).unwrap();

        // Each chunk rowid has a matching vector row with the same width.
        // sqlite-vec stores the embedding as a BLOB of raw little-endian
        // float32 values (128 * 4 = 512 bytes).
        let rows: Vec<(i64, Vec<u8>)> = {
            let mut stmt = store
                .conn
                .prepare("SELECT rowid, embedding FROM chunk_vectors ORDER BY rowid")
                .unwrap();
            let rows = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap();
            rows.collect::<Result<Vec<_>, _>>().unwrap()
        };
        assert_eq!(rows.len(), 2);
        for (i, (rowid, embedding)) in rows.iter().enumerate() {
            assert_eq!(*rowid, (i + 1) as i64);
            assert_eq!(embedding.len(), 128 * 4);
            let floats: Vec<f32> = embedding
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(floats, vectors[i]);
        }
    }
}
