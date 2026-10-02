#![warn(clippy::pedantic)]

//! Public JSON adapter; the auxiliary payload is an internal core transport.

use lode_core::fts::explanation::{FtsHit, FtsHitMapping};
use lode_core::index::explanation::{LexicalDetails, LexicalDetailsStatus};

pub(crate) fn details(details: &LexicalDetails) -> serde_json::Value {
    let matched = details.matched.as_ref();
    let phrases: Vec<_> = matched
        .into_iter()
        .flat_map(|matched| &matched.phrases)
        .map(|phrase| {
            serde_json::json!({
                "index": phrase.index,
                "query_tokens": phrase.query_tokens,
                "document_frequency": phrase.document_frequency,
                "idf": phrase.idf,
                "weighted_frequency": phrase.weighted_frequency,
                "bm25_contribution": phrase.bm25_contribution,
                "hits": phrase.hits.iter().map(hit_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::json!({
        "status": match details.status {
            LexicalDetailsStatus::Disabled => "disabled",
            LexicalDetailsStatus::EmptyQuery => "empty_query",
            LexicalDetailsStatus::NotMatched => "not_matched",
            LexicalDetailsStatus::Matched => "matched",
        },
        "in_candidate_pool": details.in_candidate_pool,
        "match_expression": details.match_expression,
        "tokenizer": {"strategy": details.strategy, "tokenize": details.tokenize},
        "bm25": matched.map(|matched| matched.bm25),
        "corpus_rows": matched.map(|matched| matched.corpus_rows),
        "document_length": matched.map(|matched| matched.document_length),
        "average_length": matched.map(|matched| matched.average_length),
        "phrases": phrases,
    })
}

fn hit_json(hit: &FtsHit) -> serde_json::Value {
    let (status, reason, span) = match &hit.mapping {
        FtsHitMapping::Mapped(span) => ("mapped", None, Some(span)),
        FtsHitMapping::Unavailable(reason) => ("unavailable", Some(reason.code()), None),
    };
    serde_json::json!({
        "column": hit.column,
        "token_offset": hit.token_offset,
        "document_tokens": hit.document_tokens,
        "mapping_status": status,
        "mapping_reason": reason,
        "start_byte": span.map(|span| span.start_byte),
        "end_byte": span.map(|span| span.end_byte),
        "surface": span.map(|span| &span.surface),
    })
}
