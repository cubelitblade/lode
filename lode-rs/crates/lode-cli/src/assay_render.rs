//! Human-readable and table output for `debug tokens`.
//!
//! The JSON payload is assembled by the command because it is part of the
//! machine contract. These renderers consume the same resolved chunk and
//! occurrence stream without changing its semantics.
#![warn(clippy::pedantic)]

use std::fmt::Write as _;

use lode_core::fts::{IndexedTerm, distinct_terms};
use lode_core::index::records::{ChunkWithRefs, FileStatus};

use crate::render::{Intent, RenderOptions};

const COMPACT_TERM_LIMIT: usize = 12;

pub(crate) fn compact(
    target: &ChunkWithRefs,
    strategy: &str,
    physical: &str,
    tokens: &[IndexedTerm],
) {
    let options = RenderOptions::stdout();
    let distinct = distinct_terms(tokens);
    let source = source_line(target);
    let digest = short_digest(&target.digest);

    crate::render::print_line(options, Intent::Info, &format!("Debug tokens · {digest}"));
    crate::render::print_line(options, Intent::Info, &source);
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("Tokenizer: strategy={strategy}, physical={physical}"),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "Terms: {} distinct · {} occurrences",
            distinct.len(),
            tokens.len()
        ),
    );

    if distinct.is_empty() {
        crate::render::print_line(options, Intent::Muted, "  (no indexed terms)");
        return;
    }

    let shown = distinct.iter().take(COMPACT_TERM_LIMIT);
    let terms = shown.map(term_label).collect::<Vec<_>>().join(" · ");
    crate::render::print_line(options, Intent::Muted, &format!("  {terms}"));
    if distinct.len() > COMPACT_TERM_LIMIT {
        crate::render::print_line(
            options,
            Intent::Muted,
            &format!(
                "  … {} more term(s); use `--view extended` for the full stream.",
                distinct.len() - COMPACT_TERM_LIMIT
            ),
        );
    }
}

pub(crate) fn extended(
    target: &ChunkWithRefs,
    strategy: &str,
    physical: &str,
    tokens: &[IndexedTerm],
) {
    let options = RenderOptions::stdout();
    let distinct = distinct_terms(tokens);

    crate::render::print_line(
        options,
        Intent::Info,
        &format!("Debug tokens · {}", target.digest),
    );
    crate::render::print_line(options, Intent::Info, "Sources:");
    for reference in &target.refs {
        crate::render::print_line(
            options,
            Intent::Muted,
            &format!(
                "  {} ({})",
                metadata_controls(&reference.path.to_native_display()),
                reference.status
            ),
        );
    }
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "Location: heading={}, page={}, seq={}",
            metadata_controls(&target.heading),
            optional_number(target.page),
            optional_number(target.seq),
        ),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("Tokenizer: strategy={strategy}, physical={physical}"),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "Counts: {} token occurrences · {} distinct terms",
            tokens.len(),
            distinct.len()
        ),
    );

    crate::render::print_line(options, Intent::Info, "Text:");
    print_raw_text(&visible_controls(&target.text));

    crate::render::print_line(options, Intent::Info, "Tokens:");
    if tokens.is_empty() {
        crate::render::print_line(options, Intent::Muted, "  (no indexed terms)");
        return;
    }
    for (index, term) in tokens.iter().enumerate() {
        crate::render::print_line(
            options,
            Intent::Muted,
            &format!("  [{index}] {}", term_label(term)),
        );
    }
}

pub(crate) fn table(target: &ChunkWithRefs, tokens: &[IndexedTerm]) {
    crate::print_tsv(&[
        "INDEX", "SURFACE", "VARIANTS", "DIGEST", "PATH", "HEADING", "PAGE", "STATE",
    ]);
    let primary = target.primary();
    let state = if chunk_stale(target) {
        "stale"
    } else {
        "fresh"
    };
    let page = target
        .page
        .map_or_else(String::new, |value| value.to_string());
    for (index, term) in tokens.iter().enumerate() {
        let index = index.to_string();
        let variants = serde_json::to_string(&term.variants).expect("variants are serializable");
        let fields = [
            index.as_str(),
            term.surface.as_str(),
            variants.as_str(),
            target.digest.as_str(),
            primary.path.as_str(),
            target.heading.as_str(),
            page.as_str(),
            state,
        ];
        crate::print_tsv(&fields);
    }
}

fn print_raw_text(text: &str) {
    crate::write_stdout(format_args!("{text}\n"));
}

pub(crate) fn source_line(target: &ChunkWithRefs) -> String {
    let primary = target.primary();
    let mut line = metadata_controls(&primary.path.to_native_display());
    if !target.heading.is_empty() {
        line.push_str(" > ");
        line.push_str(&metadata_controls(&target.heading));
    }
    if let Some(page) = target.page {
        let _ = write!(line, " (p.{page})");
    }
    if chunk_stale(target) {
        line.push_str(" [stale]");
    }
    line
}

fn chunk_stale(target: &ChunkWithRefs) -> bool {
    target
        .refs
        .iter()
        .any(|reference| reference.status == FileStatus::Stale)
}

pub(crate) fn short_digest(digest: &str) -> String {
    digest
        .strip_prefix("blake3:")
        .unwrap_or(digest)
        .chars()
        .take(12)
        .collect()
}

fn term_label(term: &IndexedTerm) -> String {
    let surface = metadata_controls(&term.surface);
    if term.variants.is_empty() {
        return surface;
    }
    let variants = term
        .variants
        .iter()
        .map(|variant| metadata_controls(variant))
        .collect::<Vec<_>>()
        .join("/");
    format!("{surface}({variants})")
}

fn visible_controls(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\n' => "\n".to_string(),
            '\t' => "\t".to_string(),
            '\r' => r"\r".to_string(),
            character if character.is_control() => format!(r"\u{{{:x}}}", character as u32),
            character => character.to_string(),
        })
        .collect()
}

fn metadata_controls(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\n' => r"\n".to_string(),
            '\t' => r"\t".to_string(),
            character => visible_controls(&character.to_string()),
        })
        .collect()
}

fn optional_number<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}
