#![warn(clippy::pedantic)]

//! Chunk and content row helpers: content addressing, GC, vector writes,
//! and chunk reads. Used by the file-reference write path and the query
//! primitives in [`super`].

use std::collections::BTreeMap;

use rusqlite::Connection;
use rusqlite::OptionalExtension;

use crate::index::records::{ChunkWithRefs, FileStatus, PathRef};
use crate::ingestion::types::Chunk;
use crate::relpath::WorkspacePath;

use std::str::FromStr;

/// Shared column list for chunk-with-path queries; positional access in
/// [`chunk_from_row`].
const CHUNK_COLUMNS: &str = "c.id, c.digest, c.text, c.heading, f.path, f.status, c.page, c.seq";

/// Convert file sizes for SQLite storage.
///
/// `size` is `u64` in the domain record but `INTEGER` (i64) in SQLite;
/// real files never approach `i64::MAX`, so the cast cannot wrap. Centralized
/// so the invariant has one home.
#[expect(
    clippy::cast_possible_wrap,
    reason = "file sizes never approach i64::MAX; SQLite stores size as INTEGER"
)]
pub(super) fn size_to_i64(size: u64) -> i64 {
    size as i64
}

/// Return `(content_id, created)` for the content with this digest.
pub(super) fn ensure_content(
    conn: &Connection,
    digest: &str,
    extractor: &str,
) -> crate::Result<(i64, bool)> {
    if !matches!(extractor, "text" | "markdown" | "doc" | "docx" | "pdf") {
        return Err(crate::Error::Store(format!(
            "unknown extractor family {extractor:?}"
        )));
    }
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM contents WHERE digest = ?1 AND extractor = ?2",
            rusqlite::params![digest, extractor],
            |row| row.get(0),
        )
        .optional()?;

    if let Some(id) = existing {
        return Ok((id, false));
    }
    conn.execute(
        "INSERT INTO contents (digest, extractor) VALUES (?1, ?2)",
        rusqlite::params![digest, extractor],
    )?;
    Ok((conn.last_insert_rowid(), true))
}

/// Drop a content row once nothing references it.
///
/// Orphanhood is derived by lookup rather than a stored refcount, so the
/// invariant holds inside the surrounding write transaction.
pub(super) fn gc_content_if_orphaned(conn: &Connection, content_id: i64) -> crate::Result<()> {
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
pub(super) fn insert_chunks(
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
        let seq = i64::from(chunk.seq);
        let page = chunk.page.map(i64::from);
        stmt.execute(rusqlite::params![
            chunk.digest,
            content_id,
            seq,
            chunk.text,
            chunk.heading,
            page,
        ])?;
        if let Some(vectors) = vectors {
            let vector = &vectors[i];
            if vector.len() != usize::try_from(dimension).unwrap_or(usize::MAX) {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "vector length is bounded by dimension (u32), which fits u32"
                )]
                let current = vector.len() as u32;
                return Err(crate::Error::DimensionMismatch {
                    stored: dimension,
                    current,
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

impl super::Store {
    /// Chunk contents for the given rowids, keyed by rowid.
    ///
    /// Each chunk carries every path referencing its content; the map
    /// iterates in rowid order. Mirrors Python's `Store.get_chunks`.
    ///
    /// # Errors
    ///
    /// Fails when the row query fails.
    ///
    /// # Panics
    ///
    /// The internal `expect` is unreachable: every new rowid group is
    /// initialized with `chunk_from_row` before refs are pushed, so the row
    /// always parses.
    pub fn get_chunks(&self, rowids: &[i64]) -> crate::Result<BTreeMap<i64, ChunkWithRefs>> {
        let conn = &self.conn;
        let mut chunks = BTreeMap::new();
        if rowids.is_empty() {
            return Ok(chunks);
        }
        let placeholders = std::iter::repeat_n("?", rowids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks c JOIN files f ON f.content_id = c.content_id \
         WHERE c.id IN ({placeholders})"
        );
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            rowids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
        let mut rows = stmt.query(params.as_slice())?;
        while let Some(row) = rows.next()? {
            let rowid: i64 = row.get(0)?;
            chunks
                .entry(rowid)
                .or_insert_with(|| chunk_from_row(row).expect("row group head"))
                .refs
                .push(path_ref_from_row(row));
        }
        Ok(chunks)
    }

    /// Chunks adjacent to a target rowid within the same section.
    ///
    /// The target itself is excluded. Neighbors are limited to the same
    /// content, heading chain, and sequence window, then returned in sequence
    /// order with all referencing paths attached. A zero radius or unknown
    /// target yields an empty result.
    ///
    /// # Errors
    ///
    /// Fails when the target or neighbor queries fail.
    ///
    /// # Panics
    ///
    /// The internal row parser is expected to succeed for rows produced by
    /// the store schema; a parser failure indicates a schema or query bug.
    pub fn get_chunk_neighbors(
        &self,
        rowid: i64,
        radius: u32,
    ) -> crate::Result<Vec<ChunkWithRefs>> {
        if radius == 0 {
            return Ok(Vec::new());
        }

        let target: Option<(i64, i64, String)> = self
            .conn
            .query_row(
                "SELECT content_id, seq, heading FROM chunks WHERE id = ?1",
                rusqlite::params![rowid],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((content_id, seq, heading)) = target else {
            return Ok(Vec::new());
        };

        let radius = i64::from(radius);
        let lower = seq.saturating_sub(radius);
        let upper = seq.saturating_add(radius);
        let sql = format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks c JOIN files f ON f.content_id = c.content_id \
             WHERE c.content_id = ?1 AND c.id != ?2 AND c.seq BETWEEN ?3 AND ?4 \
             AND c.heading = ?5 ORDER BY c.seq"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params![content_id, rowid, lower, upper, heading,])?;
        let mut grouped: BTreeMap<i64, ChunkWithRefs> = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let current_rowid: i64 = row.get(0)?;
            grouped
                .entry(current_rowid)
                .or_insert_with(|| chunk_from_row(row).expect("row group head"))
                .refs
                .push(path_ref_from_row(row));
        }

        let mut chunks: Vec<ChunkWithRefs> = grouped.into_values().collect();
        chunks.sort_by_key(|chunk| chunk.seq.unwrap_or_default());
        Ok(chunks)
    }

    /// Rowids of chunks whose digest starts with `prefix`, ordered by rowid.
    ///
    /// The prefix is the hex portion of a content address (`blake3:` already
    /// stripped). This lightweight resolver lets callers distinguish
    /// not-found from ambiguous prefixes without loading chunk bodies.
    ///
    /// # Errors
    ///
    /// Fails when the row query fails.
    pub fn find_chunk_rowids(&self, prefix: &str) -> crate::Result<Vec<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM chunks WHERE digest LIKE ?1 ORDER BY id")?;
        let rows = stmt.query_map(rusqlite::params![format!("blake3:{prefix}%")], |row| {
            row.get(0)
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Chunks whose digest starts with `prefix`, ordered by primary path then
    /// sequence.
    ///
    /// The prefix is the hex part of a content address (`blake3:` already
    /// stripped). `dig` uses this to resolve either a full digest or the short
    /// prefix `prospect` prints; each chunk carries every path referencing its
    /// content. Mirrors Python's `Store.find_chunks_by_digest`.
    ///
    /// # Errors
    ///
    /// Fails when the row query fails.
    ///
    /// # Panics
    ///
    /// The internal `expect` is unreachable — every new rowid group is
    /// initialized with `chunk_from_row` before refs are pushed.
    pub fn find_chunks_by_digest(&self, prefix: &str) -> crate::Result<Vec<ChunkWithRefs>> {
        let conn = &self.conn;
        let sql = format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks c JOIN files f ON f.content_id = c.content_id \
         WHERE c.digest LIKE ?"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params![format!("blake3:{prefix}%")])?;
        let mut grouped: BTreeMap<i64, ChunkWithRefs> = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let rowid: i64 = row.get(0)?;
            grouped
                .entry(rowid)
                .or_insert_with(|| chunk_from_row(row).expect("row group head"))
                .refs
                .push(path_ref_from_row(row));
        }
        let mut chunks: Vec<ChunkWithRefs> = grouped.into_values().collect();
        chunks.sort_by(|a, b| {
            let a_path = a.primary();
            let b_path = b.primary();
            (a_path.path.as_str(), a.seq.unwrap_or_default())
                .cmp(&(b_path.path.as_str(), b.seq.unwrap_or_default()))
        });
        Ok(chunks)
    }
}

/// Build a chunk head from one join row; refs accumulate separately.
///
/// # Panics
///
/// Never: the `expect` below is unreachable — every caller starts a new
/// group with this function before pushing refs, so the row always parses.
fn chunk_from_row(row: &rusqlite::Row<'_>) -> Result<ChunkWithRefs, rusqlite::Error> {
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "page/seq are small non-negative integers"
    )]
    Ok(ChunkWithRefs {
        digest: row.get(1)?,
        text: row.get(2)?,
        heading: row.get(3)?,
        seq: row.get::<_, Option<i64>>(7)?.map(|s| s as u32),
        page: row.get::<_, Option<i64>>(6)?.map(|p| p as u32),
        refs: Vec::new(),
    })
}

/// One reference from a join row (columns 4/5).
fn path_ref_from_row(row: &rusqlite::Row<'_>) -> PathRef {
    let path: String = row.get(4).expect("path column is NOT NULL");
    let status: String = row.get(5).expect("status column is NOT NULL");
    PathRef {
        path: WorkspacePath::from_posix(path),
        status: FileStatus::from_str(&status).unwrap_or(FileStatus::Stale),
    }
}
