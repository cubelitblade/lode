#![warn(clippy::pedantic)]

//! Typed diagnostics for the actual FTS5 MATCH query and native BM25 score.

use serde::{Deserialize, Serialize};

const ABSOLUTE_TOLERANCE: f64 = 1e-12;
const RELATIVE_TOLERANCE: f64 = 1e-12;

/// Errors specific to explicitly requested FTS diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum FtsExplanationError {
    /// SQLite rejected registration, the query, or an auxiliary callback.
    #[error("FTS explanation SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// The private auxiliary payload could not be decoded.
    #[error("FTS explanation payload error: {0}")]
    Decode(#[from] serde_json::Error),
    /// The diagnostic data does not agree with the index or native score.
    #[error("invalid FTS explanation: {0}")]
    InvalidData(&'static str),
    /// Only Lode's registered query helpers can construct a native query.
    #[error("unsupported FTS query helper: {0}")]
    InvalidHelper(&'static str),
}

/// The evaluated query and its optional match at the requested rowid.
#[derive(Debug, Clone, PartialEq)]
pub struct FtsExplanation {
    /// The exact expression evaluated on the index connection.
    pub match_expression: String,
    /// Absent for an empty expression or a target that fails the full MATCH.
    pub matched: Option<FtsMatchDetails>,
}

/// Diagnostics for a target that satisfies the full MATCH expression.
#[derive(Debug, Clone, PartialEq)]
pub struct FtsMatchDetails {
    /// The target chunk's SQLite rowid.
    pub rowid: i64,
    /// The authoritative signed score from SQLite's built-in `bm25()`.
    pub bm25: f64,
    /// Sum of the signed phrase contributions, checked against bm25.
    pub contribution_sum: f64,
    /// Number of rows in the entire FTS corpus, including nonmatching rows.
    pub corpus_rows: i64,
    /// Average number of indexed tokens per corpus row.
    pub average_length: f64,
    /// Number of indexed tokens in the target row.
    pub document_length: i32,
    /// Ordered query phrases, including repeats and unmatching OR branches.
    pub phrases: Vec<FtsPhrase>,
}

/// One FTS5 scoring phrase, which may contain several tokenizer tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FtsPhrase {
    /// Zero-based FTS5 query phrase index.
    pub index: i32,
    /// Normalized query tokens, rather than inferred natural-language words.
    pub query_tokens: Vec<String>,
    /// Corpus rows containing this phrase, respecting query column filters.
    pub document_frequency: i64,
    /// SQLite's IDF, with nonpositive values replaced by 1e-6.
    pub idf: f64,
    /// Sum of column weights over this phrase's occurrences.
    pub weighted_frequency: f64,
    /// Signed contribution to SQLite's raw BM25 score.
    pub bm25_contribution: f64,
    /// Actual occurrences, preserving overlaps and colocated variants.
    pub hits: Vec<FtsHit>,
}

/// One phrase occurrence in the indexed token stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FtsHit {
    /// Zero-based FTS column number.
    pub column: i32,
    /// Zero-based token offset where the phrase starts.
    pub token_offset: i32,
    /// Normalized document tokens. SQLite 3.46 cannot supply prefix tokens.
    pub document_tokens: Vec<Option<String>>,
    /// Original indexed-text range. Auxiliary transport never serializes it.
    #[serde(skip)]
    pub mapping: FtsHitMapping,
}

/// Original-text mapping kept separate from normalized scoring tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FtsHitMapping {
    /// A validated UTF-8 range in the indexed chunk text.
    Mapped(FtsSourceSpan),
    /// No range is claimed when mapping cannot be established.
    Unavailable(FtsMappingUnavailable),
}

impl Default for FtsHitMapping {
    fn default() -> Self {
        Self::Unavailable(FtsMappingUnavailable::NotRequested)
    }
}

/// Inclusive start and exclusive end in the indexed chunk's UTF-8 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsSourceSpan {
    /// Inclusive UTF-8 byte offset.
    pub start_byte: usize,
    /// Exclusive UTF-8 byte offset.
    pub end_byte: usize,
    /// Exact indexed-text slice; not the normalized document token.
    pub surface: String,
}

/// Reasons for withholding an original-text range without altering BM25.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtsMappingUnavailable {
    /// The caller requested raw diagnostics without original-text mapping.
    NotRequested,
    /// The actual table configuration could not be verified.
    TokenizerConfiguration,
    /// DOCUMENT tokenization failed.
    TokenizationFailed,
    /// Token flags, ranges or colocated positions were invalid.
    InvalidTokenStream,
    /// DOCUMENT positions differ from the indexed document length.
    TokenCountMismatch,
    /// The current chunk mapping supports the text column only.
    UnsupportedColumn,
    /// A hit extends beyond the actual DOCUMENT position stream.
    HitOutsideDocument,
    /// Known indexed tokens disagree with the DOCUMENT tokens at the hit.
    DocumentTokenMismatch,
}

impl FtsMappingUnavailable {
    /// Stable diagnostic reason for withholding an original-text range.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::TokenizerConfiguration => "tokenizer_configuration",
            Self::TokenizationFailed => "tokenization_failed",
            Self::InvalidTokenStream => "invalid_token_stream",
            Self::TokenCountMismatch => "token_count_mismatch",
            Self::UnsupportedColumn => "unsupported_column",
            Self::HitOutsideDocument => "hit_outside_document",
            Self::DocumentTokenMismatch => "document_token_mismatch",
        }
    }
}

/// Private transport between the auxiliary callback and the Store boundary.
/// CLI payloads must adapt the public types explicitly.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RowDetails {
    pub rowid: i64,
    pub corpus_rows: i64,
    pub average_length: f64,
    pub document_length: i32,
    pub phrases: Vec<FtsPhrase>,
}

impl RowDetails {
    pub(crate) fn into_match(
        self,
        target_rowid: i64,
        bm25: f64,
    ) -> Result<FtsMatchDetails, FtsExplanationError> {
        let sum: f64 = self
            .phrases
            .iter()
            .map(|phrase| phrase.bm25_contribution)
            .sum();
        if self.rowid != target_rowid {
            return Err(FtsExplanationError::InvalidData("target rowid mismatch"));
        }
        if !bm25_agrees(sum, bm25) {
            return Err(FtsExplanationError::InvalidData(
                "phrase contributions disagree with native BM25",
            ));
        }
        Ok(FtsMatchDetails {
            rowid: self.rowid,
            bm25,
            contribution_sum: sum,
            corpus_rows: self.corpus_rows,
            average_length: self.average_length,
            document_length: self.document_length,
            phrases: self.phrases,
        })
    }
}

pub(crate) fn bm25_agrees(value: f64, bm25: f64) -> bool {
    value.is_finite()
        && bm25.is_finite()
        && (value - bm25).abs() <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * bm25.abs()
}
