#![warn(clippy::pedantic)]

//! Opt-in FTS diagnostics. The query, BM25 and corpus statistics use the
//! index connection; no SQLite pointers or auxiliary JSON escape this layer.

use rusqlite::OptionalExtension;

use crate::fts::MatchExpr;
use crate::fts::explanation::{FtsExplanation, FtsExplanationError, RowDetails};

use super::Store;

impl Store {
    /// Explain a target's full FTS MATCH independently of candidate-pool limits.
    ///
    /// Evaluates native query helpers on this index connection, preserves empty
    /// expressions, and returns no match details when the target does not match.
    /// This is read-only and does not change sparse retrieval or ranking.
    ///
    /// # Errors
    ///
    /// Returns a sourced FTS explanation error for registration, helper, MATCH,
    /// payload or native-BM25 consistency failures. It never returns partial data.
    pub fn explain_fts(
        &self,
        expr: &MatchExpr,
        target_rowid: i64,
    ) -> crate::Result<FtsExplanation> {
        self.explain_fts_inner(expr, target_rowid)
            .map_err(Into::into)
    }

    fn explain_fts_inner(
        &self,
        expr: &MatchExpr,
        target_rowid: i64,
    ) -> Result<FtsExplanation, FtsExplanationError> {
        let match_expression = match expr {
            MatchExpr::Prebuilt(expression) => expression.clone(),
            MatchExpr::Helper { function, query } => {
                let sql = match *function {
                    "simple_query" => "SELECT simple_query(?1)",
                    "jieba_query" => "SELECT jieba_query(?1)",
                    other => return Err(FtsExplanationError::InvalidHelper(other)),
                };
                self.conn
                    .query_row(sql, [query], |row| row.get::<_, Option<String>>(0))?
                    .unwrap_or_default()
            }
        };
        let matched = if match_expression.is_empty() {
            None
        } else {
            crate::fts::auxiliary::register(&self.conn)?;
            // Both values and the auxiliary corpus scans are evaluated inside
            // one SQLite statement, so concurrent writers cannot mix snapshots.
            let row = self
                .conn
                .query_row(
                    "SELECT bm25(chunks_fts), lode_fts_details(chunks_fts)
                     FROM chunks_fts WHERE chunks_fts MATCH ?1 AND rowid = ?2",
                    rusqlite::params![match_expression, target_rowid],
                    |row| Ok((row.get::<_, f64>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()?;
            match row {
                Some((bm25, json)) => {
                    let details: RowDetails = serde_json::from_str(&json)?;
                    Some(details.into_match(target_rowid, bm25)?)
                }
                None => None,
            }
        };
        Ok(FtsExplanation {
            match_expression,
            matched,
        })
    }
}

#[cfg(test)]
mod tests;
