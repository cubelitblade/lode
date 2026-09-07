#![warn(clippy::pedantic)]

//! Pluggable ranking operators: per-source `Norm` + cross-source `Fusion`.
//!
//! Mirrors `src/lode/index/ranking.py`. Python defines `Norm`/`Fusion` as
//! protocols (runtime duck typing); here they are closed enums selected by
//! config, so match sites stay exhaustive and no dynamic dispatch is needed.
//!
//! Score maps are keyed by chunk rowid and use `BTreeMap` for deterministic
//! iteration (rowid ascending). Python relies on dict insertion order (SQL
//! row order) as the tie-break for equal scores; rowid order is at least as
//! deterministic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::explanation::{
    EvidenceBlock, FormulaComponents, FusionExplanation, RankingFactor, RetrievalStatus,
    SourceExplanation,
};
use super::records::Source;

/// Per-source score maps before fusion, keyed by source then chunk rowid.
pub type PreparedScores = BTreeMap<Source, BTreeMap<i64, f64>>;

/// Per-source score transform, applied before fusion.
///
/// The transform must be monotonic so it never changes the source's internal
/// ranking (RRF relies on this to skip normalization safely).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Norm {
    /// Min-max normalize to [0, 1]; a single value maps to 1.0.
    MinMax,
    /// Softmax over scores scaled by `temperature`; higher flattens toward
    /// uniform, lower sharpens toward the top score.
    #[serde(rename = "softmax")]
    Softmax {
        /// Scaling temperature; 1.0 by default.
        temperature: f64,
    },
}

impl Norm {
    /// Operator name used in explanation output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::MinMax => "min-max",
            Self::Softmax { .. } => "softmax",
        }
    }

    /// Normalize one source's raw scores to a comparable domain.
    #[must_use]
    pub fn normalize(self, scores: &BTreeMap<i64, f64>) -> BTreeMap<i64, f64> {
        match self {
            Self::MinMax => minmax_normalize(scores),
            Self::Softmax { temperature } => softmax_normalize(scores, temperature),
        }
    }
}

/// Min-max normalize to [0, 1]; a single value maps to 1.0.
fn minmax_normalize(scores: &BTreeMap<i64, f64>) -> BTreeMap<i64, f64> {
    let Some(&lo) = scores.values().next() else {
        return BTreeMap::new();
    };
    let (lo, hi) = scores
        .values()
        .fold((lo, lo), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = hi - lo;
    if span == 0.0 {
        return scores.keys().map(|&rowid| (rowid, 1.0)).collect();
    }
    scores
        .iter()
        .map(|(&rowid, &score)| (rowid, (score - lo) / span))
        .collect()
}

/// Softmax over scores scaled by `temperature`.
///
/// The transform is monotonic in each score, so it preserves the source's
/// internal ranking.
fn softmax_normalize(scores: &BTreeMap<i64, f64>, temperature: f64) -> BTreeMap<i64, f64> {
    if scores.is_empty() {
        return BTreeMap::new();
    }
    let scaled = scores.values().map(|&score| score / temperature);
    // Subtract the max before exp to keep the sum finite for large inputs.
    let max_scaled = scaled.clone().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.map(|value| (value - max_scaled).exp()).collect();
    let total: f64 = exps.iter().sum();
    scores
        .keys()
        .zip(exps)
        .map(|(&rowid, exp)| (rowid, exp / total))
        .collect()
}

/// Cross-source merge of prepared per-source scores into one combined score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Fusion {
    /// Weighted sum of per-source scores: `sum(w_s * prepared[s][rowid])`.
    #[serde(rename = "linear")]
    Linear {
        /// Fusion weight per source; missing sources weigh zero.
        weights: BTreeMap<Source, f64>,
    },
    /// Reciprocal rank fusion: `sum(1 / (k + rank_s(rowid)))`.
    ///
    /// Ranks are derived from the prepared scores (descending). Because
    /// min-max and softmax are monotonic, the rank is identical whether
    /// computed on raw or normalized scores — which is why RRF can skip
    /// normalization entirely.
    #[serde(rename = "rrf")]
    Rrf {
        /// Rank offset; larger k flattens the reciprocal curve.
        k: u64,
    },
}

impl Fusion {
    /// Operator name used in explanation output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear { .. } => "linear",
            Self::Rrf { .. } => "rrf",
        }
    }

    /// Merge the per-source prepared scores into one combined score map.
    #[must_use]
    pub fn fuse(self, prepared: &PreparedScores) -> BTreeMap<i64, f64> {
        let mut combined: BTreeMap<i64, f64> = BTreeMap::new();
        match self {
            Self::Linear { weights } => {
                for (source, scores) in prepared {
                    let weight = weights.get(source).copied().unwrap_or(0.0);
                    for (&rowid, &score) in scores {
                        *combined.entry(rowid).or_default() += weight * score;
                    }
                }
            }
            Self::Rrf { k } => {
                for scores in prepared.values() {
                    let mut ranked: Vec<(&i64, &f64)> = scores.iter().collect();
                    // `total_cmp` is a total order, so no NaN ambiguity and
                    // no panic path (unlike `partial_cmp().expect(...)`).
                    ranked.sort_by(|a, b| b.1.total_cmp(a.1));
                    for (position, (rowid, _)) in ranked.into_iter().enumerate() {
                        let position = u64::try_from(position + 1).unwrap_or(u64::MAX);
                        *combined.entry(*rowid).or_default() += 1.0 / rank_as_f64(k + position);
                    }
                }
            }
        }
        combined
    }

    /// Render-ready explanation of how this fusion produced `combined`.
    ///
    /// The render layer consumes this without branching on concrete operator
    /// types. `norm` is provided so Linear can include the normalization
    /// name in evidence blocks; RRF ignores it.
    #[must_use]
    pub fn explain(
        self,
        sources: &BTreeMap<Source, SourceExplanation>,
        combined: f64,
        norm: Option<Norm>,
    ) -> FusionExplanation {
        match self {
            Self::Linear { weights } => linear_explain(&weights, sources, combined, norm),
            Self::Rrf { k } => rrf_explain(k, sources, combined),
        }
    }
}

/// The retrieval path a query actually runs, frozen at assembly time.
///
/// `norm` is `None` when the fusion ranks by position (RRF), in which case
/// normalization is skipped. `fusion` carries its own parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalPlan {
    /// Per-source norm, `None` when fusion skips normalization.
    pub norm: Option<Norm>,
    /// Cross-source fusion operator.
    pub fusion: Fusion,
}

/// Convert a rank/offset to `f64`.
///
/// Pool ranks and RRF offsets are small integers, far below `f64`'s 52-bit
/// mantissa, so the cast cannot lose meaning; the lint is noise here.
#[expect(
    clippy::cast_precision_loss,
    reason = "ranks and offsets are small integers, far below f64 mantissa precision"
)]
fn rank_as_f64(value: u64) -> f64 {
    value as f64
}

fn linear_explain(
    weights: &BTreeMap<Source, f64>,
    sources: &BTreeMap<Source, SourceExplanation>,
    combined: f64,
    norm: Option<Norm>,
) -> FusionExplanation {
    let norm_name = norm.map(|n| n.name().to_string());
    let mut ranking_factors = BTreeMap::new();
    let mut evidence = BTreeMap::new();
    let mut symbolic_parts: Vec<String> = Vec::new();
    let mut value_parts: Vec<String> = Vec::new();

    for (name, source) in sources {
        if source.status != RetrievalStatus::Matched {
            continue;
        }
        let weight = weights.get(name).copied().unwrap_or(0.0);
        let prepared = source.prepared_score.unwrap_or(0.0);
        let contribution = prepared * weight;
        ranking_factors.insert(
            *name,
            RankingFactor {
                metric: "contribution".into(),
                value: contribution,
            },
        );
        evidence.insert(
            *name,
            EvidenceBlock {
                rank: source.pool_rank,
                raw_score: source.raw_score,
                prepared_score: source.prepared_score,
                normalization: norm_name.clone(),
                weight: Some(weight),
                contribution: Some(contribution),
            },
        );
        symbolic_parts.push(format!("{name} × {weight}"));
        value_parts.push(format!("{prepared:.4} × {weight}"));
    }

    let formula = FormulaComponents {
        method_label: "Linear".into(),
        symbolic_terms: symbolic_parts.join(" + "),
        value_terms: value_parts.join(" + "),
        result: combined,
        missing_note: None,
    };
    FusionExplanation {
        ranking_factors,
        evidence,
        formula,
    }
}

fn rrf_explain(
    k: u64,
    sources: &BTreeMap<Source, SourceExplanation>,
    combined: f64,
) -> FusionExplanation {
    let mut ranking_factors = BTreeMap::new();
    let mut symbolic_parts: Vec<String> = Vec::new();
    let mut value_parts: Vec<String> = Vec::new();

    for (name, source) in sources {
        if source.status != RetrievalStatus::Matched {
            continue;
        }
        let rank = source.pool_rank.unwrap_or(0);
        ranking_factors.insert(
            *name,
            RankingFactor {
                metric: "rank".into(),
                value: rank_as_f64(rank as u64),
            },
        );
        symbolic_parts.push(format!("1 ÷ ({k} + {name}_rank)"));
        value_parts.push(format!("1 ÷ {}", k + rank as u64));
    }

    let mut missing: Vec<String> = sources
        .iter()
        .filter(|(_, source)| source.status != RetrievalStatus::Matched)
        .map(|(name, source)| format!("{name}: {}", source.status.label()))
        .collect();
    missing.sort();

    let formula = FormulaComponents {
        method_label: format!("RRF (k={k})"),
        symbolic_terms: symbolic_parts.join(" + "),
        value_terms: value_parts.join(" + "),
        result: combined,
        missing_note: (!missing.is_empty()).then(|| missing.join(", ")),
    };
    FusionExplanation {
        ranking_factors,
        evidence: BTreeMap::new(),
        formula,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores<const N: usize>(entries: [(i64, f64); N]) -> BTreeMap<i64, f64> {
        entries.into_iter().collect()
    }

    fn weights<const N: usize>(entries: [(Source, f64); N]) -> BTreeMap<Source, f64> {
        entries.into_iter().collect()
    }

    fn prepared(entries: impl IntoIterator<Item = (Source, Vec<(i64, f64)>)>) -> PreparedScores {
        entries
            .into_iter()
            .map(|(source, pairs)| (source, pairs.into_iter().collect()))
            .collect()
    }

    #[test]
    fn minmax_normalizes_to_unit_interval() {
        let norm = Norm::MinMax;
        let result = norm.normalize(&scores([(1, 0.0), (2, 5.0), (3, 10.0)]));
        assert_eq!(result, scores([(1, 0.0), (2, 0.5), (3, 1.0)]));
    }

    #[test]
    fn minmax_single_value_maps_to_one() {
        assert_eq!(
            Norm::MinMax.normalize(&scores([(1, 3.0)])),
            scores([(1, 1.0)])
        );
    }

    #[test]
    fn minmax_empty_is_empty() {
        assert_eq!(Norm::MinMax.normalize(&BTreeMap::new()), BTreeMap::new());
    }

    #[test]
    fn minmax_flat_scores_map_to_one() {
        // All scores equal -> span is zero -> every value maps to 1.0.
        assert_eq!(
            Norm::MinMax.normalize(&scores([(1, 2.0), (2, 2.0)])),
            scores([(1, 1.0), (2, 1.0)])
        );
    }

    #[test]
    fn softmax_sums_to_one() {
        let norm = Norm::Softmax { temperature: 1.0 };
        let result = norm.normalize(&scores([(1, 1.0), (2, 2.0), (3, 3.0)]));
        assert!((result.values().sum::<f64>() - 1.0).abs() < 1e-9);
        // Higher raw score -> higher normalized score (monotonic).
        assert!(result[&3] > result[&2] && result[&2] > result[&1]);
    }

    #[test]
    fn softmax_temperature_flattens() {
        let hot = Norm::Softmax { temperature: 10.0 }.normalize(&scores([(1, 1.0), (2, 2.0)]));
        let cold = Norm::Softmax { temperature: 0.1 }.normalize(&scores([(1, 1.0), (2, 2.0)]));
        // Higher temperature spreads scores toward uniform.
        assert!((hot[&2] - hot[&1]) < (cold[&2] - cold[&1]));
    }

    #[test]
    fn softmax_empty_is_empty() {
        assert_eq!(
            Norm::Softmax { temperature: 1.0 }.normalize(&BTreeMap::new()),
            BTreeMap::new()
        );
    }

    #[test]
    fn linear_fusion_weighted_sum() {
        let fusion = Fusion::Linear {
            weights: weights([(Source::Semantic, 0.7), (Source::Lexical, 0.3)]),
        };
        let result = fusion.fuse(&prepared([
            (Source::Semantic, vec![(1, 1.0), (2, 0.5)]),
            (Source::Lexical, vec![(2, 1.0), (3, 0.5)]),
        ]));
        assert!((result[&1] - 0.7).abs() < 1e-12);
        assert!((result[&2] - (0.7 * 0.5 + 0.3)).abs() < 1e-12);
        assert!((result[&3] - 0.3 * 0.5).abs() < 1e-12);
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn linear_fusion_unknown_source_weight_zero() {
        let fusion = Fusion::Linear {
            weights: weights([(Source::Semantic, 1.0)]),
        };
        let result = fusion.fuse(&prepared([
            (Source::Semantic, vec![]),
            (Source::Lexical, vec![(1, 5.0)]),
        ]));
        assert_close(result[&1], 0.0);
    }

    #[test]
    fn rrf_fusion_reciprocal_rank() {
        let fusion = Fusion::Rrf { k: 60 };
        let result = fusion.fuse(&prepared([
            (Source::Semantic, vec![(1, 10.0), (2, 5.0)]),
            (Source::Lexical, vec![(2, 9.0), (3, 1.0)]),
        ]));
        // semantic ranks: 1 -> pos1, 2 -> pos2; lexical ranks: 2 -> pos1, 3 -> pos2.
        assert!((result[&1] - 1.0 / 61.0).abs() < 1e-12);
        assert!((result[&2] - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-12);
        assert!((result[&3] - 1.0 / 62.0).abs() < 1e-12);
    }

    #[test]
    fn rrf_fusion_uses_custom_k() {
        let fusion = Fusion::Rrf { k: 10 };
        let result = fusion.fuse(&prepared([
            (Source::Semantic, vec![(1, 1.0)]),
            (Source::Lexical, vec![]),
        ]));
        assert!((result[&1] - 1.0 / 11.0).abs() < 1e-12);
    }

    fn matched(raw: f64, prepared: f64, pool_rank: usize) -> SourceExplanation {
        SourceExplanation {
            status: RetrievalStatus::Matched,
            pool_size: 40,
            raw_score: Some(raw),
            prepared_score: Some(prepared),
            pool_rank: Some(pool_rank),
        }
    }

    fn sources() -> BTreeMap<Source, SourceExplanation> {
        [
            (Source::Semantic, matched(0.9, 1.0, 1)),
            (Source::Lexical, matched(-5.0, 0.5, 3)),
        ]
        .into_iter()
        .collect()
    }

    fn linear_fusion() -> Fusion {
        Fusion::Linear {
            weights: weights([(Source::Semantic, 0.7), (Source::Lexical, 0.3)]),
        }
    }

    #[test]
    fn linear_explain_ranking_factors_are_contributions() {
        let expl = linear_fusion().explain(&sources(), 0.85, Some(Norm::MinMax));
        assert_eq!(
            expl.ranking_factors[&Source::Semantic].metric,
            "contribution"
        );
        assert!((expl.ranking_factors[&Source::Semantic].value - 0.7).abs() < 1e-12);
        assert!((expl.ranking_factors[&Source::Lexical].value - 0.15).abs() < 1e-12);
    }

    #[test]
    fn linear_explain_evidence_carries_norm_and_weight() {
        let expl = linear_fusion().explain(&sources(), 0.85, Some(Norm::MinMax));
        let block = &expl.evidence[&Source::Semantic];
        assert_eq!(block.rank, Some(1));
        assert!(block.raw_score.is_some_and(|v| (v - 0.9).abs() < 1e-12));
        assert!(
            block
                .prepared_score
                .is_some_and(|v| (v - 1.0).abs() < 1e-12)
        );
        assert_eq!(block.normalization.as_deref(), Some("min-max"));
        assert!(block.weight.is_some_and(|v| (v - 0.7).abs() < 1e-12));
        assert!(block.contribution.is_some_and(|v| (v - 0.7).abs() < 1e-12));
    }

    #[test]
    fn linear_explain_formula_substitutes_numbers() {
        let expl = linear_fusion().explain(&sources(), 0.85, Some(Norm::MinMax));
        assert_eq!(expl.formula.method_label, "Linear");
        assert_eq!(
            expl.formula.symbolic_terms,
            "semantic × 0.7 + lexical × 0.3"
        );
        assert_eq!(expl.formula.value_terms, "1.0000 × 0.7 + 0.5000 × 0.3");
        assert_close(expl.formula.result, 0.85);
        assert_eq!(expl.formula.missing_note, None);
    }

    #[test]
    fn linear_explain_skips_non_matched_sources() {
        let mut sources = sources();
        sources.insert(
            Source::Lexical,
            SourceExplanation::new(RetrievalStatus::Disabled, 0),
        );
        let expl = linear_fusion().explain(&sources, 0.7, Some(Norm::MinMax));
        assert_eq!(expl.ranking_factors.len(), 1);
        assert!(expl.ranking_factors.contains_key(&Source::Semantic));
        assert_eq!(expl.formula.symbolic_terms, "semantic × 0.7");
    }

    #[test]
    fn rrf_explain_ranking_factors_are_ranks() {
        let expl = Fusion::Rrf { k: 60 }.explain(&sources(), 0.05, None);
        assert_eq!(expl.ranking_factors[&Source::Semantic].metric, "rank");
        assert_close(expl.ranking_factors[&Source::Semantic].value, 1.0);
        assert_close(expl.ranking_factors[&Source::Lexical].value, 3.0);
    }

    #[test]
    fn rrf_explain_evidence_is_empty() {
        let expl = Fusion::Rrf { k: 60 }.explain(&sources(), 0.05, None);
        assert!(expl.evidence.is_empty());
    }

    #[test]
    fn rrf_explain_formula_uses_ranks() {
        let expl = Fusion::Rrf { k: 60 }.explain(&sources(), 0.05, None);
        assert_eq!(expl.formula.method_label, "RRF (k=60)");
        assert_eq!(
            expl.formula.symbolic_terms,
            "1 ÷ (60 + semantic_rank) + 1 ÷ (60 + lexical_rank)"
        );
        assert_eq!(expl.formula.value_terms, "1 ÷ 61 + 1 ÷ 63");
        assert_close(expl.formula.result, 0.05);
        assert_eq!(expl.formula.missing_note, None);
    }

    #[test]
    fn rrf_explain_missing_note_lists_non_matched() {
        let mut sources = sources();
        sources.insert(
            Source::Lexical,
            SourceExplanation::new(RetrievalStatus::Empty, 0),
        );
        let expl = Fusion::Rrf { k: 60 }.explain(&sources, 1.0 / 61.0, None);
        assert_eq!(expl.ranking_factors.len(), 1);
        assert!(expl.ranking_factors.contains_key(&Source::Semantic));
        assert_eq!(
            expl.formula.missing_note.as_deref(),
            Some("lexical: no results")
        );
    }
}
