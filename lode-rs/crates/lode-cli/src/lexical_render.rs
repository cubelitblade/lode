#![warn(clippy::pedantic)]

//! Human-readable FTS diagnostics, using the same typed data as JSON.

use lode_core::fts::explanation::{FtsMatchDetails, FtsPhrase};
use lode_core::index::explanation::{LexicalDetails, LexicalDetailsStatus};

use crate::render::{Intent, RenderOptions};
use crate::why_render::{format_score, metadata_controls};

pub(crate) fn extended(details: &LexicalDetails) {
    let options = RenderOptions::stdout();
    render(details, |intent, text| {
        crate::render::print_line(options, intent, text);
    });
}

fn render(details: &LexicalDetails, mut print: impl FnMut(Intent, &str)) {
    print(Intent::Info, "FTS details (raw BM25):");
    print(
        Intent::Muted,
        &format!(
            "  tokenizer: strategy={} · physical={}",
            metadata_controls(&details.strategy),
            metadata_controls(&details.tokenize),
        ),
    );
    let expression = details.match_expression.as_ref().map_or_else(
        || "— (disabled)".to_owned(),
        |expression| {
            if expression.is_empty() {
                "\"\" (empty)".to_owned()
            } else {
                metadata_controls(expression)
            }
        },
    );
    print(Intent::Muted, &format!("  MATCH: {expression}"));
    let status = match details.status {
        LexicalDetailsStatus::Disabled => "disabled",
        LexicalDetailsStatus::EmptyQuery => "empty_query",
        LexicalDetailsStatus::NotMatched => "not_matched",
        LexicalDetailsStatus::Matched => "matched",
    };
    print(
        Intent::Info,
        &format!(
            "  full MATCH status: {status} · in lexical candidate pool: {}",
            if details.in_candidate_pool {
                "yes"
            } else {
                "no"
            },
        ),
    );
    let Some(matched) = &details.matched else {
        let reason = match details.status {
            LexicalDetailsStatus::Disabled => {
                "Lexical retrieval is disabled; MATCH and phrase statistics were skipped."
            }
            LexicalDetailsStatus::EmptyQuery => {
                "The lexical expression is empty; MATCH and phrase statistics were skipped."
            }
            LexicalDetailsStatus::NotMatched => {
                "The target does not satisfy the full MATCH; no phrase statistics are available."
            }
            LexicalDetailsStatus::Matched => "No phrase statistics are available.",
        };
        print(Intent::Muted, &format!("  {reason}"));
        return;
    };
    if !details.in_candidate_pool {
        print(
            Intent::Info,
            "  This BM25 score did not participate in fusion: the target is outside the lexical candidate pool.",
        );
    }
    render_matched(matched, &mut print);
}

fn render_matched(matched: &FtsMatchDetails, print: &mut impl FnMut(Intent, &str)) {
    print(
        Intent::Muted,
        "  Each phrase is one FTS5 scoring unit and may contain multiple tokens.",
    );
    print(
        Intent::Muted,
        "  Contributions explain signed raw BM25 only; they do not attribute normalized or fused scores (including RRF).",
    );
    print(
        Intent::Info,
        &format!(
            "  raw BM25 (display rounded): {}",
            format_score(Some(matched.bm25)),
        ),
    );
    print(
        Intent::Muted,
        &format!(
            "  corpus rows: {} · document length: {} · average length (display rounded): {}",
            matched.corpus_rows,
            matched.document_length,
            format_score(Some(matched.average_length)),
        ),
    );
    print(
        Intent::Muted,
        "  Hit columns and token offsets are zero based; token offsets are not text byte offsets.",
    );
    for phrase in &matched.phrases {
        render_phrase(phrase, print);
    }
}

fn render_phrase(phrase: &FtsPhrase, print: &mut impl FnMut(Intent, &str)) {
    print(
        Intent::Info,
        &format!(
            "  phrase #{}: query tokens={} · hits={} · BM25 contribution (display rounded)={}",
            phrase.index,
            visible_json(&serde_json::json!(&phrase.query_tokens)),
            phrase.hits.len(),
            format_score(Some(phrase.bm25_contribution)),
        ),
    );
    print(
        Intent::Muted,
        &format!(
            "    document frequency: {} · IDF (display rounded): {} · weighted frequency: {}",
            phrase.document_frequency,
            format_score(Some(phrase.idf)),
            format_score(Some(phrase.weighted_frequency)),
        ),
    );
    for hit in &phrase.hits {
        let unknown = if hit.document_tokens.iter().any(Option::is_none) {
            " (null = unknown document token)"
        } else {
            ""
        };
        print(
            Intent::Muted,
            &format!(
                "    hit: column={} · token offset={} · document tokens={}{}",
                hit.column,
                hit.token_offset,
                visible_json(&serde_json::json!(&hit.document_tokens)),
                unknown,
            ),
        );
    }
}

fn visible_json(value: &serde_json::Value) -> String {
    metadata_controls(&value.to_string())
}

#[cfg(test)]
mod tests;
