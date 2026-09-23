//! Human-readable, table, and JSON output for `assay how QUERY`.

#![warn(clippy::pedantic)]

use crate::render::{Intent, RenderOptions};

pub(crate) fn compact(
    query: &str,
    strategy: &str,
    physical: &str,
    helper: Option<&str>,
    match_expression: &str,
) {
    let options = RenderOptions::stdout();
    crate::render::print_line(
        options,
        Intent::Info,
        &format!("Assay how · {}", visible_controls(query)),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("Strategy: {strategy} · physical tokenizer: {physical}"),
    );
    crate::render::print_line(options, Intent::Muted, &format!("Rule: {}", rule(strategy)));
    if let Some(helper) = helper {
        crate::render::print_line(options, Intent::Muted, &format!("Helper: {helper}"));
    }
    print_expression(options, match_expression);
}

pub(crate) fn extended(
    query: &str,
    strategy: &str,
    physical: &str,
    helper: Option<&str>,
    match_expression: &str,
) {
    let options = RenderOptions::stdout();
    crate::render::print_line(options, Intent::Info, "Query explanation:");
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("  Query: {}", visible_controls(query)),
    );
    crate::render::print_line(options, Intent::Muted, &format!("  Strategy: {strategy}"));
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("  Physical tokenizer: {physical}"),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("  Helper: {}", helper.unwrap_or("")),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("  Rule: {}", rule(strategy)),
    );
    crate::render::print_line(options, Intent::Info, "  MATCH expression:");
    if match_expression.is_empty() {
        crate::render::print_line(
            options,
            Intent::Muted,
            "    (no lexical match expression was generated)",
        );
    } else {
        crate::render::print_line(
            options,
            Intent::Muted,
            &format!("    {}", visible_controls(match_expression)),
        );
    }
}

pub(crate) fn table(
    query: &str,
    strategy: &str,
    physical: &str,
    helper: Option<&str>,
    match_expression: &str,
) {
    crate::print_tsv(&["QUERY", "STRATEGY", "TOKENIZER", "HELPER", "MATCH"]);
    crate::print_tsv(&[
        query,
        strategy,
        physical,
        helper.unwrap_or(""),
        match_expression,
    ]);
}

pub(crate) fn json(
    query: &str,
    strategy: &str,
    physical: &str,
    helper: Option<&str>,
    match_expression: &str,
) {
    let payload = serde_json::json!({
        "query": query,
        "tokenizer": {
            "strategy": strategy,
            "tokenize": physical,
        },
        "helper": helper,
        "match_expression": match_expression,
    });
    crate::write_stdout(format_args!(
        "{}",
        serde_json::to_string_pretty(&payload).expect("query explanation is serializable")
    ));
}

fn print_expression(options: RenderOptions, match_expression: &str) {
    if match_expression.is_empty() {
        crate::render::print_line(
            options,
            Intent::Muted,
            "MATCH: (no lexical match expression was generated)",
        );
    } else {
        crate::render::print_line(
            options,
            Intent::Muted,
            &format!("MATCH: {}", visible_controls(match_expression)),
        );
    }
}

fn rule(strategy: &str) -> &'static str {
    match strategy {
        "unicode61" => {
            "word fragments are quoted and joined with OR; SQLite applies its unicode61 tokenizer"
        }
        "trigram" => "three-character windows are quoted and joined with OR",
        "simple" => {
            "the native helper joins query units with AND and can add pinyin and ASCII prefixes"
        }
        "jieba" => {
            "the native helper segments query text into dictionary words over the simple index"
        }
        _ => "the configured lexical strategy constructs the MATCH expression",
    }
}

fn visible_controls(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\r' => r"\r".to_string(),
            '\n' => r"\n".to_string(),
            '\t' => r"\t".to_string(),
            character if character.is_control() => format!(r"\u{{{:x}}}", character as u32),
            character => character.to_string(),
        })
        .collect()
}
