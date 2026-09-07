#![warn(clippy::pedantic)]

//! FTS5 tokenizers and MATCH query construction.
//!
//! Renamed from `lexical`: the module is entirely about SQLite FTS5
//! integration, matching the `FtsConfig` / `fts.strategy` naming. The
//! Python side still calls it `lode.lexical`; align when the Python tree
//! retires.
//!
//! The [`MatchExpr`] enum models the two query shapes FTS5 supports so the
//! store can stay a pure SQL boundary while callers build the expression
//! per the configured strategy.

/// A FTS5 MATCH expression for one query, in one of the two supported forms.
///
/// Python keys the branch off `strategy.uses_helper`; this closed enum is the
/// exhaustive equivalent. `sparse_search` consumes it and picks the SQL text
/// accordingly — the user query never reaches SQL as raw text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchExpr {
    /// A pre-built MATCH expression (unicode61/trigram strategies): tokens
    /// quoted and `OR`-joined by [`match_query`]. Bound as a parameter.
    Prebuilt(String),
    /// A native tokenizer helper function (`simple_query`); the raw query is
    /// bound as a parameter and the helper evaluates it inside FTS5.
    Helper {
        /// The helper function name, e.g. `simple_query`.
        function: &'static str,
        /// The raw user query.
        query: String,
    },
}

/// Build the MATCH expression for `query` under the tokenizer named by
/// `strategy`.
///
/// The full strategy registry lands with the lexical step; the two built-in
/// SQLite tokenizers are complete here.
///
/// # Errors
///
/// # Errors
///
/// Fails for the native `simple`/`jieba` strategies, which need the
/// extension library the Rust rewrite does not link yet.
pub fn match_query(query: &str, strategy: &str) -> crate::Result<MatchExpr> {
    match strategy {
        "unicode61" => Ok(MatchExpr::Prebuilt(unicode61_query(query))),
        "trigram" => Ok(MatchExpr::Prebuilt(trigram_query(query))),
        other => Err(crate::Error::Fts(format!(
            "tokenizer {other:?} needs the native extension, which the Rust \
             rewrite does not implement yet; set fts.strategy = \"unicode61\""
        ))),
    }
}

/// Word-ish token runs for the plain (unicode61-style) query.
///
/// Each token is quoted so punctuation in user queries cannot break the
/// MATCH syntax; mirrors Python's `\w+` findall.
fn word_tokens(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// The unicode61-style query: tokens quoted, `OR`-joined.
fn unicode61_query(text: &str) -> String {
    word_tokens(text)
        .iter()
        .map(|token| format!("\"{token}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// The trigram query: raw 3-grams quoted, `OR`-joined (CJK-friendly
/// substring matching). A shorter-than-3-chars input yields no grams, which
/// matches Python's behaviour of an empty expression.
fn trigram_query(text: &str) -> String {
    text.chars()
        .collect::<Vec<char>>()
        .windows(3)
        .map(|window| window.iter().collect::<String>())
        .map(|gram| format!("\"{gram}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode61_splits_on_word_boundaries() {
        // Punctuation is dropped, not kept inside tokens.
        let expr = match_query("ore, vein!", "unicode61").unwrap();
        assert_eq!(expr, MatchExpr::Prebuilt("\"ore\" OR \"vein\"".to_string()));
    }

    #[test]
    fn unicode61_keeps_underscore() {
        let expr = match_query("a_b c", "unicode61").unwrap();
        assert_eq!(expr, MatchExpr::Prebuilt("\"a_b\" OR \"c\"".to_string()));
    }

    #[test]
    fn trigram_slices_into_grams() {
        let expr = match_query("矿石探矿", "trigram").unwrap();
        assert_eq!(
            expr,
            MatchExpr::Prebuilt("\"矿石探\" OR \"石探矿\"".to_string())
        );
    }

    #[test]
    fn trigram_short_input_yields_empty() {
        let expr = match_query("ab", "trigram").unwrap();
        assert_eq!(expr, MatchExpr::Prebuilt(String::new()));
    }

    #[test]
    fn native_tokenizer_is_refused_with_hint() {
        let err = match_query("x", "simple").unwrap_err();
        assert!(err.to_string().contains("native extension"));
        assert!(err.to_string().contains("unicode61"));
    }
}
