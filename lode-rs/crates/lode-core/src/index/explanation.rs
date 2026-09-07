#![warn(clippy::pedantic)]

//! Score explanation data structures.
//!
//! Pure data types consumed by the `assay why` render layer. The [`Fusion`]
//! operators (see `super::ranking`) return a [`FusionExplanation`] from their
//! `explain()` methods; the render layer formats it without branching on
//! concrete operator types.
//!
//! Mirrors `src/lode/index/explanation.py`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::records::Source;

/// One retrieval source's participation in a query.
///
/// The four states are mutually exclusive and derivable from the scoring
/// pass: `Disabled` (never queried — e.g. a zero linear weight), `Empty`
/// (queried but returned no candidates at all), `NotRetrieved` (the pool has
/// candidates but not this chunk), `Matched` (this chunk is in the pool with
/// scores).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalStatus {
    /// Never queried, e.g. a zero linear weight.
    Disabled,
    /// Queried but returned no candidates at all.
    Empty,
    /// The pool has candidates but not this chunk.
    NotRetrieved,
    /// This chunk is in the pool with scores.
    Matched,
}

impl RetrievalStatus {
    /// Human-readable label for this status.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Empty => "no results",
            Self::NotRetrieved => "not retrieved",
            Self::Matched => "matched",
        }
    }
}

/// Per-source facts for one chunk in one query.
///
/// Invariant: when `status` is [`RetrievalStatus::Matched`], `raw_score`,
/// `prepared_score` and `pool_rank` are all `Some`; for every other status
/// they are all `None`. `pool_size` is the source's actual candidate count
/// (0 unless the source was queried).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SourceExplanation {
    /// Participation status of this source for the chunk.
    pub status: RetrievalStatus,
    /// The source's actual candidate count (0 unless it was queried).
    pub pool_size: usize,
    /// Un-normalized score (cosine for semantic, BM25 for lexical).
    pub raw_score: Option<f64>,
    /// Score after the plan's norm, or `raw_score` when the norm is skipped.
    pub prepared_score: Option<f64>,
    /// 1-based rank within the source pool by raw score, descending.
    pub pool_rank: Option<usize>,
}

impl SourceExplanation {
    /// A source that did not produce a candidate pool for this chunk.
    ///
    /// Covers every status except [`RetrievalStatus::Matched`]; fill the
    /// score fields directly for matched sources.
    #[must_use]
    pub fn new(status: RetrievalStatus, pool_size: usize) -> Self {
        Self {
            status,
            pool_size,
            raw_score: None,
            prepared_score: None,
            pool_rank: None,
        }
    }
}

/// One ranking factor for one source in the [`FusionExplanation`].
///
/// The source identity is the mapping key in `FusionExplanation`'s
/// `ranking_factors`, not a field here — the key is the single source of
/// truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingFactor {
    /// What this factor measures: `"contribution"` (weighted score) or
    /// `"rank"`.
    pub metric: String,
    /// The numeric value to display.
    pub value: f64,
}

/// Per-source evidence detail for one MATCHED source.
///
/// All fields are optional: RRF fills only `rank` and `raw_score` (no
/// normalization, no weight/contribution). Linear fills all fields. The
/// render layer decides formatting based on which fields are `None`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EvidenceBlock {
    /// 1-based rank within the source pool.
    pub rank: Option<usize>,
    /// Un-normalized score.
    pub raw_score: Option<f64>,
    /// Score after normalization.
    pub prepared_score: Option<f64>,
    /// Normalization operator name, e.g. `"softmax"`.
    pub normalization: Option<String>,
    /// Fusion weight of the source (linear only).
    pub weight: Option<f64>,
    /// `prepared_score * weight` (linear only).
    pub contribution: Option<f64>,
}

/// Self-contained fusion formula display data.
///
/// The render layer formats this into the Fusion section without any
/// operator-type checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormulaComponents {
    /// e.g. `"RRF (k=60)"` or `"Linear"`.
    pub method_label: String,
    /// e.g. `"1 ÷ (60 + semantic_rank)"` or `"semantic × 0.7 + lexical × 0.3"`.
    pub symbolic_terms: String,
    /// e.g. `"1 ÷ 61 + 1 ÷ 63"` or `"0.9000 × 0.7 + 0.5000 × 0.3"`.
    pub value_terms: String,
    /// The combined score after fusion.
    pub result: f64,
    /// Optional note for non-matched sources, e.g. `"lexical: no results"`.
    pub missing_note: Option<String>,
}

/// Complete render-ready explanation from a fusion operator's `explain()`.
///
/// Consumed by the `assay why` render layer without branching on concrete
/// operator types — all fusion-specific logic lives in the operator's
/// `explain()` implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionExplanation {
    /// One entry per MATCHED source.
    pub ranking_factors: BTreeMap<Source, RankingFactor>,
    /// One entry per MATCHED source. Empty for RRF (no normalization/weight
    /// info).
    pub evidence: BTreeMap<Source, EvidenceBlock>,
    /// The fusion formula with actual numbers substituted.
    pub formula: FormulaComponents,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_labels_match_python() {
        assert_eq!(RetrievalStatus::Disabled.label(), "disabled");
        assert_eq!(RetrievalStatus::Empty.label(), "no results");
        assert_eq!(RetrievalStatus::NotRetrieved.label(), "not retrieved");
        assert_eq!(RetrievalStatus::Matched.label(), "matched");
    }

    #[test]
    fn status_serializes_as_python_values() {
        let json = serde_json::to_string(&RetrievalStatus::NotRetrieved).unwrap();
        assert_eq!(json, r#""not_retrieved""#);
        let back: RetrievalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RetrievalStatus::NotRetrieved);
    }

    #[test]
    fn source_explanation_defaults_none_scores() {
        let e = SourceExplanation::new(RetrievalStatus::Empty, 0);
        assert_eq!(e.status, RetrievalStatus::Empty);
        assert_eq!(e.pool_size, 0);
        assert_eq!(e.raw_score, None);
        assert_eq!(e.prepared_score, None);
        assert_eq!(e.pool_rank, None);
    }

    #[test]
    fn fusion_explanation_json_roundtrip() {
        let mut factors = BTreeMap::new();
        factors.insert(
            Source::Semantic,
            RankingFactor {
                metric: "contribution".into(),
                value: 0.63,
            },
        );
        let explanation = FusionExplanation {
            ranking_factors: factors,
            evidence: BTreeMap::new(),
            formula: FormulaComponents {
                method_label: "Linear".into(),
                symbolic_terms: "semantic × 0.7".into(),
                value_terms: "0.9000 × 0.7".into(),
                result: 0.63,
                missing_note: Some("lexical: no results".into()),
            },
        };
        let json = serde_json::to_string(&explanation).unwrap();
        let back: FusionExplanation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, explanation);
        // Source keys serialize as Python's source names.
        assert!(json.contains(r#""semantic""#));
    }
}
