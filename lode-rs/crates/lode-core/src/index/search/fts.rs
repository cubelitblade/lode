#![warn(clippy::pedantic)]

//! Compose opt-in MATCH diagnostics with the unchanged retrieval scores.

use crate::embeddings::base::Embedder;
use crate::fts::MatchExpr;
use crate::fts::explanation::{FtsExplanationError, FtsMatchDetails, bm25_agrees};
use crate::index::explanation::{
    LexicalDetails, LexicalDetailsStatus, ScoreExplanationWithFts, SourceExplanation,
};
use crate::index::ranking::RetrievalPlan;
use crate::index::records::Source;
use crate::index::store::Store;

use super::{explanation_from_candidates, score_candidates_for_query, source_enabled};

/// Explain scores and full MATCH details using a single index read snapshot.
///
/// The target read fixes the snapshot before embedding, so a concurrent WAL
/// writer cannot mix candidate scores, target text and diagnostic statistics.
///
/// # Errors
///
/// Returns scoring errors unchanged. Diagnostic or consistency failures return
/// a sourced FTS explanation error, never a partial explanation.
pub fn explain_with_fts(
    store: &Store,
    embedder: &dyn Embedder,
    query: &str,
    rowid: i64,
    plan: &RetrievalPlan,
    top_k: u32,
) -> crate::Result<ScoreExplanationWithFts> {
    store.with_read_snapshot(|store| {
        let Some(chunk) = store.get_chunks(&[rowid])?.remove(&rowid) else {
            return Err(crate::Error::Store(format!("no chunk with rowid {rowid}")));
        };
        let strategy = store.query_tokenizer();
        let tokenize = crate::index::schema::physical_tokenizer(strategy).ok_or(
            FtsExplanationError::InvalidData("unknown physical tokenizer"),
        )?;
        let expression = if source_enabled(plan, Source::Lexical) {
            Some(store.resolve_fts_query(&crate::fts::match_query(query, strategy)?)?)
        } else {
            None
        };
        let candidates =
            score_candidates_for_query(store, embedder, query, plan, top_k, expression.as_deref())?;
        let in_candidate_pool = candidates.lexical_raw.contains_key(&rowid);
        let explanation = explanation_from_candidates(chunk, &candidates, rowid, plan, top_k);
        let matched = match &expression {
            Some(expression) => {
                store
                    .explain_fts(&MatchExpr::Prebuilt(expression.clone()), rowid)?
                    .matched
            }
            None => None,
        };
        validate_pool_score(explanation.sources.get(&Source::Lexical), matched.as_ref())?;
        let status = match expression.as_deref() {
            None => LexicalDetailsStatus::Disabled,
            Some("") => LexicalDetailsStatus::EmptyQuery,
            Some(_) if matched.is_some() => LexicalDetailsStatus::Matched,
            Some(_) => LexicalDetailsStatus::NotMatched,
        };
        Ok(ScoreExplanationWithFts {
            explanation,
            lexical_details: LexicalDetails {
                status,
                in_candidate_pool,
                strategy: strategy.into(),
                tokenize: tokenize.into(),
                match_expression: expression,
                matched,
            },
        })
    })
}

fn validate_pool_score(
    source: Option<&SourceExplanation>,
    matched: Option<&FtsMatchDetails>,
) -> Result<(), FtsExplanationError> {
    let Some(raw_score) = source.and_then(|source| source.raw_score) else {
        return Ok(());
    };
    let Some(matched) = matched else {
        return Err(FtsExplanationError::InvalidData(
            "candidate fails full MATCH",
        ));
    };
    if !bm25_agrees(raw_score, matched.bm25) {
        return Err(FtsExplanationError::InvalidData("candidate BM25 mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
