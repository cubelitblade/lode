#![warn(clippy::pedantic)]

//! SQLite-backed store: the [`Store`] type, its lifecycle, and the file-record
//! write path.
//!
//! Split by concern: `meta` owns the `meta` table header and connection
//! pragmas; `chunks` owns content-addressed row helpers and vector writes.
//! Query primitives (dense/sparse search, chunk reads) live in sibling
//! modules as they land.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use rusqlite::Connection;
use rusqlite::OptionalExtension;

use crate::index::records::{FileRecord, FileStatus};
use crate::index::schema;
use crate::ingestion::types::Chunk;
use crate::relpath::WorkspacePath;

mod chunks;
mod meta;
mod query;

use self::chunks::{ensure_content, gc_content_if_orphaned, insert_chunks, size_to_i64};
use self::meta::{IndexMeta, configure_connection, read_meta, write_meta};

/// Schema version; must match the database to open it.
pub use crate::index::schema::SCHEMA_VERSION;

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
    /// Creation needs the embedder's model id and vector dimension (the
    /// dimension sizes the vec0 table) plus the FTS5 tokenizer; all three
    /// are recorded in `meta` and validated on later opens. Fails if the
    /// database exists but is incompatible.
    ///
    /// # Errors
    ///
    /// Fails when creation or validation fails; see [`Store::create`] and
    /// [`Store::open_existing`].
    pub fn open(
        path: &Path,
        model_id: &str,
        dimension: u32,
        tokenizer: &str,
    ) -> crate::Result<Self> {
        if path.is_file() {
            Self::open_existing(path)
        } else {
            Self::create(path, model_id, dimension, tokenizer)
        }
    }

    /// Create a fresh index database with the full schema.
    fn create(path: &Path, model_id: &str, dimension: u32, tokenizer: &str) -> crate::Result<Self> {
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
        write_meta(&conn, model_id, dimension, tokenizer)?;
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
    /// returns a ready-to-use store.
    ///
    /// # Errors
    ///
    /// Fails when the database does not exist, the schema version is
    /// incompatible, or required metadata keys are missing.
    pub fn open_existing(path: &Path) -> crate::Result<Self> {
        if !path.is_file() {
            return Err(crate::Error::Store(format!(
                "database does not exist: {}",
                path.display()
            )));
        }

        // vec0 is a sqlite-vec virtual table; register the extension as an
        // auto-extension so every connection (this one and later opens) can
        // use it. Registration is process-global, so guard it with `Once`,
        // and it must happen *before* the connection is opened: the module
        // lookup happens per connection at open time.
        register_vec_extension();

        let conn = Connection::open(path)?;
        configure_connection(&conn)?;
        let meta = read_meta(&conn)?;

        Ok(Self {
            path: path.to_path_buf(),
            conn,
            meta,
        })
    }

    /// Archive an existing index database out of the way so a fresh one can
    /// be built in its place.
    ///
    /// Moves the database and its WAL companion files (`-wal`, `-shm`) to
    /// sibling `.bak` paths, tolerating companions that are absent. Because
    /// the CLI is the only writer, archiving by rename is sound: the caller
    /// drops any open connections first, then reopening via [`Store::open`]
    /// lands on the create path and rebuilds a pristine index.
    ///
    /// Returns whether the primary database existed (i.e. anything was
    /// archived). Deliberately does not validate the embedder — the caller
    /// owns that — mirroring Python, where compatibility checks precede
    /// destruction so a mis-configured run leaves the old index intact.
    ///
    /// # Errors
    ///
    /// Fails when the archive rename fails.
    pub fn reset(path: &Path) -> crate::Result<bool> {
        let existed = path.is_file();
        for (tail, bak_tail) in [("", ".bak"), ("-wal", "-wal.bak"), ("-shm", "-shm.bak")] {
            let src = Self::appended(path, tail);
            let dst = Self::appended(path, bak_tail);
            if matches!(src.try_exists(), Ok(true)) {
                std::fs::rename(src, dst)?;
            }
        }
        Ok(existed)
    }

    /// Append `suffix` to a path's file-name portion (keeping any extension).
    fn appended(path: &Path, suffix: &str) -> PathBuf {
        let mut os = path.as_os_str().to_owned();
        os.push(suffix);
        PathBuf::from(os)
    }

    /// The stored index metadata.
    pub fn meta(&self) -> &IndexMeta {
        &self.meta
    }

    /// All indexed paths with their content metadata, sorted by path.
    ///
    /// # Errors
    ///
    /// Fails when the row query fails.
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
    ///
    /// # Errors
    ///
    /// Fails when the update statement fails.
    pub fn mark_stale(&mut self, path: &WorkspacePath) -> crate::Result<()> {
        self.conn.execute(
            "UPDATE files SET status = ?1 WHERE path = ?2",
            rusqlite::params![FileStatus::Stale.as_str(), path.as_str()],
        )?;
        Ok(())
    }

    /// Metadata for one indexed path, or `None` when it is not indexed.
    ///
    /// # Errors
    ///
    /// Fails when the row query fails.
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
    ///
    /// # Errors
    ///
    /// Fails when the lookup, upsert, or GC statements fail.
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
                size_to_i64(record.size),
                record.status.as_str(),
            ],
        )?;

        if let Some(old_id) = previous
            && old_id != content_id
        {
            gc_content_if_orphaned(&tx, old_id)?;
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
    ///
    /// # Errors
    ///
    /// Fails when chunk and vector counts differ, a vector width differs
    /// from the index dimension, or any statement fails.
    pub fn replace_file(
        &mut self,
        record: &FileRecord,
        chunks: &[Chunk],
        vectors: Option<&[Vec<f32>]>,
    ) -> crate::Result<bool> {
        if let Some(vectors) = vectors
            && chunks.len() != vectors.len()
        {
            return Err(crate::Error::Store(format!(
                "got {} chunks but {} vectors",
                chunks.len(),
                vectors.len()
            )));
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
                size_to_i64(record.size),
                record.status.as_str(),
            ],
        )?;

        // Write chunks and vectors only when content is newly created.
        if created {
            insert_chunks(&tx, content_id, chunks, vectors, self.meta.dimension)?;
        }

        if let Some(old_id) = previous
            && old_id != content_id
        {
            gc_content_if_orphaned(&tx, old_id)?;
        }

        tx.commit()?;
        Ok(created)
    }

    /// Delete a path reference; drop its content when this was the last one.
    ///
    /// # Errors
    ///
    /// Fails when the lookup, delete, or GC statements fail.
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

/// Register the sqlite-vec extension as a SQLite auto-extension.
///
/// `sqlite3_auto_extension` is process-global and affects every connection
/// opened afterwards, so it runs exactly once via `Once`. The `transmute`
/// mirrors sqlite-vec's own test: the entry point is a plain `extern "C"`
/// function, but rusqlite's binding types it with the extension API
/// signature.
fn register_vec_extension() {
    static REGISTER_VEC: std::sync::Once = std::sync::Once::new();
    REGISTER_VEC.call_once(|| {
        // The transmute target type is fixed by rusqlite's binding, not by
        // this call site, so clippy cannot infer it.
        #[expect(
            clippy::missing_transmute_annotations,
            reason = "target type is fixed by rusqlite's binding, not this call site"
        )]
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
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

        let store = Store::open(&db, "test-model", 512, "unicode61").unwrap();

        // Metadata header is recorded.
        assert_eq!(store.meta().schema_version, SCHEMA_VERSION.to_string());
        assert_eq!(store.meta().model_id, "test-model");
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

        Store::open(&db, "test-model", 256, "unicode61").unwrap();

        let reopened = Store::open_existing(&db).unwrap();
        assert_eq!(reopened.meta().dimension, 256);
        assert_eq!(reopened.meta().tokenizer, "unicode61");
        assert_eq!(reopened.meta().model_id, "test-model");
    }

    #[test]
    fn open_existing_registers_vec_module() {
        // Regression: `no such module: vec0` when reopening an existing
        // index and writing vectors — the auto-extension must be registered
        // before `open_existing` opens its connection, not only on the
        // create path.
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();
        let rec = make_record("a.txt", "blake3:seed", 1.0, 10);
        store
            .replace_file(
                &rec,
                &make_chunks("blake3:seed", 1),
                Some(&make_vectors(1, 128)),
            )
            .unwrap();
        drop(store);

        let mut reopened = Store::open_existing(&db).unwrap();
        let rec = make_record("b.txt", "blake3:other", 1.0, 10);
        let wrote = reopened
            .replace_file(
                &rec,
                &make_chunks("blake3:other", 1),
                Some(&make_vectors(1, 128)),
            )
            .unwrap();
        assert!(wrote);
    }

    #[test]
    fn reset_archives_the_db_out_of_the_way() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join(".lode").join("index.db");

        // Populate a database so the archive provably carried data away.
        {
            let mut store = Store::open(&db, "test-model", 768, "unicode61").unwrap();
            let rec = FileRecord {
                path: WorkspacePath::from_posix("doc.md"),
                digest: "abc".into(),
                mtime: 1_710_000_000.0,
                size: 777,
                status: FileStatus::Fresh,
            };
            store.reference_file(&rec).unwrap();
        }

        // Archiving happened and cleared the primary path.
        assert!(Store::reset(&db).unwrap());
        assert!(!db.exists());

        let bak = dir.path().join(".lode").join("index.db.bak");
        assert!(bak.is_file(), "archive file should exist");

        // Opening again rebuilds a pristine index carrying the new metadata.
        let rebuilt = Store::open(&db, "second-model", 896, "unicode61").unwrap();
        assert_eq!(rebuilt.meta().model_id, "second-model");
        assert_eq!(rebuilt.meta().dimension, 896);
        assert_eq!(rebuilt.list_files().unwrap().len(), 0);
    }

    #[test]
    fn reset_is_a_no_op_when_no_db_exists() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("absent.db");

        assert!(!Store::reset(&db).unwrap());
        assert!(!db.exists());
    }

    #[test]
    fn open_rejects_zero_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let err = Store::open(&db, "test-model", 0, "simple").unwrap_err();
        assert!(err.to_string().contains("dimension must be positive"));
    }

    #[test]
    fn open_rejects_unknown_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let err = Store::open(&db, "test-model", 512, "bogus").unwrap_err();
        assert!(err.to_string().contains("unknown tokenizer"));
    }

    #[test]
    fn open_rejects_native_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");

        let err = Store::open(&db, "test-model", 512, "simple").unwrap_err();
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
            .map(|i| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "test chunk counts are tiny"
                )]
                let seq = i as u32;
                Chunk {
                    digest: digest.to_string(),
                    text: format!("chunk {i}"),
                    seq,
                    heading: String::new(),
                    page: None,
                }
            })
            .collect()
    }

    /// `n` vectors of `dimension` width, each entry equal to its index.
    fn make_vectors(n: usize, dimension: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "test indices are tiny; exact value is irrelevant"
                )]
                let value = i as f32;
                vec![value; dimension]
            })
            .collect()
    }

    #[test]
    fn reference_file_new_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

        let rec = make_record("a.txt", "blake3:aaa", 1.0, 100);
        assert!(!store.reference_file(&rec).unwrap());
        // Nothing was written — the content was not indexed yet.
        assert!(store.list_files().unwrap().is_empty());
    }

    #[test]
    fn reference_file_reuses_existing_content() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        assert!((files[0].mtime - 2.0).abs() < f64::EPSILON);

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
        let mut store = Store::open(&db, "test-model", 128, "unicode61").unwrap();

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
            assert_eq!(*rowid, i64::try_from(i + 1).unwrap_or_default());
            assert_eq!(embedding.len(), 128 * 4);
            let floats: Vec<f32> = embedding
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            assert_eq!(floats, vectors[i]);
        }
    }

    // -- get_chunks / find_chunks_by_digest / dense_search / sparse_search --

    use crate::fts::MatchExpr;

    /// Seed one file with `n` chunks and vectors; returns the store.
    fn seeded_store(digest: &str, texts: &[&str]) -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, "test-model", 4, "unicode61").unwrap();
        let chunks: Vec<Chunk> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| Chunk {
                digest: digest.to_string(),
                text: text.to_string(),
                #[expect(clippy::cast_possible_truncation, reason = "tiny chunk count")]
                seq: i as u32,
                heading: String::new(),
                page: None,
            })
            .collect();
        let vectors: Vec<Vec<f32>> = texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "tiny test indices; exact float value is irrelevant"
                )]
                let value = i as f32;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "tiny test chunk counts; exact float value is irrelevant"
                )]
                let len = texts.len() as f32;
                vec![value / len; 4]
            })
            .collect();
        let rec = make_record("doc.txt", digest, 1.0, 10);
        store.replace_file(&rec, &chunks, Some(&vectors)).unwrap();
        (dir, store)
    }

    #[test]
    fn get_chunks_joins_refs_and_keys_by_rowid() {
        let (_dir, store) = seeded_store("blake3:seed", &["alpha", "beta", "gamma"]);

        // rowids are 1..=3 for a single replace_file.
        let chunks = store.get_chunks(&[1, 3]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks.contains_key(&1) && chunks.contains_key(&3));
        let chunk = &chunks[&1];
        assert_eq!(chunk.text, "alpha");
        assert_eq!(chunk.digest, "blake3:seed");
        assert_eq!(chunk.refs.len(), 1);
        assert_eq!(chunk.refs[0].path.as_str(), "doc.txt");
        assert_eq!(chunk.refs[0].status, FileStatus::Fresh);
        assert_eq!(chunk.seq, Some(0));
    }

    #[test]
    fn get_chunks_empty_rowids_is_empty_map() {
        let (_dir, store) = seeded_store("blake3:kkk", &["x"]);
        assert!(store.get_chunks(&[]).unwrap().is_empty());
    }

    #[test]
    fn get_chunks_carries_every_referencing_path() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, "test-model", 4, "unicode61").unwrap();

        // First file creates the content; second path shares it.
        let rec1 = make_record("a.txt", "blake3:shared", 1.0, 10);
        store
            .replace_file(&rec1, &make_chunks("blake3:shared", 1), None)
            .unwrap();
        let rec2 = make_record("b.txt", "blake3:shared", 2.0, 10);
        store.reference_file(&rec2).unwrap();

        let chunks = store.get_chunks(&[1]).unwrap();
        let chunk = &chunks[&1];
        let mut paths: Vec<&str> = chunk.refs.iter().map(|r| r.path.as_str()).collect();
        paths.sort_unstable();
        assert_eq!(paths, ["a.txt", "b.txt"]);
    }

    #[test]
    fn find_chunks_by_digest_resolves_prefix_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.db");
        let mut store = Store::open(&db, "test-model", 4, "unicode61").unwrap();

        // Two contents whose digests share a prefix, each on its own path.
        let rec1 = make_record("z.txt", "blake3:dead0001", 1.0, 10);
        let rec2 = make_record("a.txt", "blake3:dead0002", 1.0, 10);
        store
            .replace_file(&rec1, &make_chunks("blake3:dead0001", 2), None)
            .unwrap();
        store
            .replace_file(&rec2, &make_chunks("blake3:dead0002", 2), None)
            .unwrap();

        let chunks = store.find_chunks_by_digest("dead").unwrap();
        assert_eq!(chunks.len(), 4);
        // Sorted by primary path then seq: a.txt chunks first, in seq order.
        assert_eq!(chunks[0].text, "chunk 0");
        assert_eq!(chunks[0].refs[0].path.as_str(), "a.txt");
        assert_eq!(chunks[2].refs[0].path.as_str(), "z.txt");

        // Full-digest lookup works too.
        let one = store.find_chunks_by_digest("dead0002").unwrap();
        assert_eq!(one.len(), 2);
        assert_eq!(one[0].digest, "blake3:dead0002");

        // Unknown prefix yields nothing.
        assert!(store.find_chunks_by_digest("beef").unwrap().is_empty());
    }

    #[test]
    fn dense_search_returns_nearest_first() {
        let (_dir, store) = seeded_store("blake3:vec", &["zero", "one", "two"]);

        // Unit vectors: chunk i has value i/3; query matches chunk 1 best.
        let query = vec![0.25_f32, 0.25, 0.25, 0.25];
        let hits = store.dense_search(&query, 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert!(hits[0].rowid == 2 || hits[0].rowid == 1 || hits[0].rowid == 3);
        // Distances are ascending.
        assert!(hits[0].distance <= hits[1].distance);
        assert!(hits[1].distance <= hits[2].distance);
    }

    #[test]
    fn dense_search_maps_dimension_mismatch() {
        let (_dir, store) = seeded_store("blake3:vec", &["x"]);

        let wrong = vec![0.0_f32; 8];
        let err = store.dense_search(&wrong, 1).unwrap_err();
        match err {
            crate::Error::DimensionMismatch { stored, current } => {
                assert_eq!(stored, 4);
                assert_eq!(current, 8);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn sparse_search_matches_prebuilt_expression() {
        let (_dir, store) = seeded_store(
            "blake3:fts",
            &["ore vein mining", "surface mining", "unrelated"],
        );

        let expr = MatchExpr::Prebuilt("\"ore\" OR \"vein\"".to_string());
        let hits = store.sparse_search(&expr, 5).unwrap();
        assert!(!hits.is_empty());
        // Best-first: descending score (BM25 is negative).
        assert!(hits[0].score >= hits[hits.len() - 1].score);
        // The matching chunk is rowid 1 (first chunk of the content).
        assert_eq!(hits[0].rowid, 1);

        // Non-matching query yields nothing.
        let misses = store
            .sparse_search(&MatchExpr::Prebuilt("\"quartz\"".to_string()), 5)
            .unwrap();
        assert!(misses.is_empty());
    }

    #[test]
    fn sparse_search_zero_k_is_empty() {
        let (_dir, store) = seeded_store("blake3:z", &["x"]);
        let expr = MatchExpr::Prebuilt("\"x\"".to_string());
        assert!(store.sparse_search(&expr, 0).unwrap().is_empty());
        assert!(store.dense_search(&[0.0; 4], 0).unwrap().is_empty());
    }
}
