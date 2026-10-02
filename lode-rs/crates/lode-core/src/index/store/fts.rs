#![warn(clippy::pedantic)]

//! Opt-in FTS diagnostics. The query, BM25 and corpus statistics use the
//! index connection; no SQLite pointers or auxiliary JSON escape this layer.

use rusqlite::OptionalExtension;

use crate::fts::MatchExpr;
use crate::fts::explanation::{
    FtsExplanation, FtsExplanationError, FtsMappingUnavailable, FtsMatchDetails, RowDetails,
};

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

    pub(crate) fn resolve_fts_query(
        &self,
        expr: &MatchExpr,
    ) -> Result<String, FtsExplanationError> {
        Ok(match expr {
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
        })
    }

    pub(crate) fn map_fts_hits(&self, details: &mut FtsMatchDetails, text: &str) {
        let tokens = self.document_tokens(text);
        crate::fts::mapping::apply(details, text, tokens);
    }

    fn document_tokens(
        &self,
        text: &str,
    ) -> Result<Vec<lode_simple_native::Token>, FtsMappingUnavailable> {
        let physical = crate::index::schema::physical_tokenizer(self.query_tokenizer())
            .ok_or(FtsMappingUnavailable::TokenizerConfiguration)?;
        let sql: String = self
            .conn
            .query_row(
                "SELECT sql FROM sqlite_schema WHERE name = 'chunks_fts' AND type = 'table'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| FtsMappingUnavailable::TokenizerConfiguration)?;
        if !crate::fts::mapping::same_definition(
            &sql,
            &crate::index::schema::fts_table_sql(physical),
        ) {
            return Err(FtsMappingUnavailable::TokenizerConfiguration);
        }
        match physical {
            "unicode61" => {
                crate::fts::auxiliary::builtin_document_tokens(&self.conn, c"unicode61", text)
                    .map_err(|_| FtsMappingUnavailable::TokenizationFailed)
            }
            "trigram" => {
                crate::fts::auxiliary::builtin_document_tokens(&self.conn, c"trigram", text)
                    .map_err(|_| FtsMappingUnavailable::TokenizationFailed)
            }
            "simple" => lode_simple_native::tokenize_document(&self.conn, text)
                .map_err(|_| FtsMappingUnavailable::TokenizationFailed),
            _ => Err(FtsMappingUnavailable::TokenizerConfiguration),
        }
    }

    fn explain_fts_inner(
        &self,
        expr: &MatchExpr,
        target_rowid: i64,
    ) -> Result<FtsExplanation, FtsExplanationError> {
        let match_expression = self.resolve_fts_query(expr)?;
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

#[cfg(test)]
mod mapping_tests;
