#![warn(clippy::pedantic)]

//! Retrieval primitives: dense kNN over the vec0 table and sparse BM25 over
//! the FTS5 table. Pure SQL boundaries — the hybrid fusion lives in
//! `index::search` (later step).

use crate::fts::MatchExpr;
use crate::index::records::{DenseMatch, SparseMatch};

use super::Store;

/// SQL for a vec0 k-nearest-neighbor query.
///
/// vec0's MATCH syntax takes the limit as a *literal*, not a bound parameter,
/// so `k` must be interpolated into the statement. `u32` bounds the
/// interpolation to a plain non-negative integer (never a string or a
/// fragment), keeping the injection surface closed — mirrors Python's
/// `int(k)` guard.
fn dense_knn_sql(k: u32) -> String {
    format!(
        "SELECT rowid, distance FROM chunk_vectors \
         WHERE embedding MATCH ?1 AND k = {k} ORDER BY distance"
    )
}

impl Store {
    /// k nearest neighbors as `(rowid, distance)`, nearest first.
    ///
    /// The query vector is serialized as a JSON array (what sqlite-vec
    /// expects) and bound as a parameter. A vector whose width differs from
    /// the index dimension surfaces SQLite's "Dimension mismatch" error,
    /// which is mapped to [`crate::Error::DimensionMismatch`], mirroring
    /// Python's `DimensionMismatchError` recovery path.
    ///
    /// # Errors
    ///
    /// Fails on query errors, including the dimension mismatch mapped to
    /// [`crate::Error::DimensionMismatch`].
    pub fn dense_search(&self, vector: &[f32], k: u32) -> crate::Result<Vec<DenseMatch>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let sql = dense_knn_sql(k);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params![serde_json::to_string(vector).map_err(
            |e| crate::Error::Store(format!("could not serialize query vector: {e}"))
        )?])?;

        // The vec0 MATCH check surfaces at the first step, not at prepare,
        // so the mapping must cover both the query call and the row loop.
        let map_err = |err: rusqlite::Error| {
            let text = err.to_string();
            if text.contains("Dimension mismatch")
                // sqlite-vec spells it either way depending on the check
                || text.contains("dimension mismatch")
            {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "query vector length is bounded by the dimension the embedder produces"
                )]
                let current = vector.len() as u32;
                crate::Error::DimensionMismatch {
                    stored: self.meta.dimension,
                    current,
                }
            } else {
                crate::Error::from(err)
            }
        };

        let mut hits = Vec::new();
        while let Some(row) = rows.next().map_err(&map_err)? {
            hits.push(DenseMatch {
                rowid: row.get(0)?,
                distance: row.get(1)?,
            });
        }
        Ok(hits)
    }

    /// k best FTS5 matches for the MATCH expression `expr`, best first.
    ///
    /// SQLite BM25 scores are negative and closer to zero means better, so
    /// best-first is descending score. Mirrors Python's `Store.sparse_search`
    /// with the strategy branch lifted into [`MatchExpr`]: `Prebuilt` binds
    /// the expression as a parameter, `Helper` interpolates the helper
    /// function call into the SQL text and binds the raw query.
    ///
    /// # Errors
    ///
    /// Fails when the row query fails.
    pub fn sparse_search(&self, expr: &MatchExpr, k: u32) -> crate::Result<Vec<SparseMatch>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut stmt = match expr {
            MatchExpr::Prebuilt(_) => self.conn.prepare(
                "SELECT rowid, bm25(chunks_fts) FROM chunks_fts \
                 WHERE chunks_fts MATCH ?1 \
                 ORDER BY bm25(chunks_fts) DESC LIMIT ?2",
            )?,
            MatchExpr::Helper { function, .. } => self.conn.prepare(&format!(
                "SELECT rowid, bm25(chunks_fts) FROM chunks_fts \
                 WHERE chunks_fts MATCH {function}(?) \
                 ORDER BY bm25(chunks_fts) DESC LIMIT ?2"
            ))?,
        };
        let query_params: Vec<&dyn rusqlite::ToSql> = match expr {
            MatchExpr::Prebuilt(text) => {
                vec![text, &k]
            }
            MatchExpr::Helper { query, .. } => {
                vec![query, &k]
            }
        };
        let mut hits = Vec::new();
        let mut rows = stmt.query(query_params.as_slice())?;
        while let Some(row) = rows.next()? {
            hits.push(SparseMatch {
                rowid: row.get(0)?,
                score: row.get(1)?,
            });
        }
        Ok(hits)
    }
}

/// Validate that a kNN helper renders the expected SQL shape.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_sql_interpolates_k_literal() {
        assert_eq!(
            dense_knn_sql(8),
            "SELECT rowid, distance FROM chunk_vectors \
             WHERE embedding MATCH ?1 AND k = 8 ORDER BY distance"
        );
    }
}
