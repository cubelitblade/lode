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

/// One primary FTS5 term and any tokens emitted at the same position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTerm {
    /// The primary token emitted by the document tokenizer.
    pub surface: String,
    /// Additional tokens emitted with `FTS5_TOKEN_COLOCATED`.
    pub variants: Vec<String>,
}

/// Tokenize text through the native document tokenizer and preserve its
/// colocated-token structure for callers such as future lexical inspection.
///
/// # Errors
///
/// Returns the native tokenizer or FTS5 callback error if the connection has
/// not been registered or emits invalid token data.
pub fn tokenize_preview(
    conn: &rusqlite::Connection,
    text: &str,
) -> crate::Result<Vec<IndexedTerm>> {
    let tokens = lode_simple_native::tokenize_document(conn, text)?;
    let mut terms: Vec<IndexedTerm> = Vec::new();
    for token in tokens {
        if token.flags & lode_simple_native::TOKEN_COLOCATED != 0 {
            let Some(previous) = terms.last_mut() else {
                return Err(crate::Error::Fts(
                    "native tokenizer emitted a colocated token first".to_string(),
                ));
            };
            previous.variants.push(token.text);
        } else {
            terms.push(IndexedTerm {
                surface: token.text,
                variants: Vec::new(),
            });
        }
    }
    Ok(terms)
}

/// Preview the index-side token stream for one configured FTS strategy.
///
/// The preview uses a private in-memory SQLite connection so it observes the
/// same tokenizer that the index uses without touching the workspace index.
/// `simple` and `jieba` share the physical native `simple` tokenizer; `jieba`
/// only changes query construction, so the document token stream is the same
/// for both strategies.
///
/// # Errors
///
/// Returns an error when the strategy is unknown, SQLite cannot create the
/// preview objects, or the native tokenizer reports a failure.
pub fn preview_index_terms(strategy: &str, text: &str) -> crate::Result<Vec<IndexedTerm>> {
    match strategy {
        "simple" | "jieba" => {
            let conn = rusqlite::Connection::open_in_memory()?;
            lode_simple_native::register_connection(&conn, false)?;
            tokenize_preview(&conn, text)
        }
        "unicode61" | "trigram" => preview_builtin_terms(strategy, text),
        other => Err(crate::Error::Fts(format!(
            "unknown tokenizer strategy {other:?}"
        ))),
    }
}

/// Preview one of SQLite's built-in FTS5 document tokenizers.
fn preview_builtin_terms(strategy: &str, text: &str) -> crate::Result<Vec<IndexedTerm>> {
    let conn = rusqlite::Connection::open_in_memory()?;
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE preview USING fts5(text, tokenize = '{strategy}');
         CREATE VIRTUAL TABLE preview_vocab USING fts5vocab(preview, instance);"
    ))?;
    conn.execute(
        "INSERT INTO preview(rowid, text) VALUES (1, ?1)",
        rusqlite::params![text],
    )?;

    let mut statement = conn.prepare(
        "SELECT term
         FROM preview_vocab
         WHERE doc = 1 AND col = 'text'
         ORDER BY offset",
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    let mut terms = Vec::new();
    for row in rows {
        terms.push(IndexedTerm {
            surface: row?,
            variants: Vec::new(),
        });
    }
    Ok(terms)
}

/// Merge repeated primary terms while retaining first-seen order.
#[must_use]
pub fn distinct_terms(terms: &[IndexedTerm]) -> Vec<IndexedTerm> {
    let mut merged: Vec<IndexedTerm> = Vec::new();
    for term in terms {
        if let Some(existing) = merged.iter_mut().find(|item| item.surface == term.surface) {
            for variant in &term.variants {
                if !existing.variants.contains(variant) {
                    existing.variants.push(variant.clone());
                }
            }
        } else {
            merged.push(term.clone());
        }
    }
    merged
}

/// Build the MATCH expression for `query` under the tokenizer named by
/// `strategy`.
///
/// Native strategies return a helper expression so the raw query remains a
/// bound SQL parameter. Both strategies use the physical `simple` tokenizer;
/// `jieba` only changes query construction.
///
/// # Errors
///
/// Returns an error when `strategy` is not one of the configured tokenizer
/// names.
pub fn match_query(query: &str, strategy: &str) -> crate::Result<MatchExpr> {
    match strategy {
        "unicode61" => Ok(MatchExpr::Prebuilt(unicode61_query(query))),
        "trigram" => Ok(MatchExpr::Prebuilt(trigram_query(query))),
        "simple" => Ok(MatchExpr::Helper {
            function: "simple_query",
            query: query.to_string(),
        }),
        "jieba" => Ok(MatchExpr::Helper {
            function: "jieba_query",
            query: query.to_string(),
        }),
        other => Err(crate::Error::Fts(format!(
            "unknown tokenizer strategy {other:?}"
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
    use rusqlite::Connection;

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
    fn native_tokenizers_use_their_query_helpers() {
        assert_eq!(
            match_query("x", "simple").unwrap(),
            MatchExpr::Helper {
                function: "simple_query",
                query: "x".to_string(),
            }
        );
        assert_eq!(
            match_query("矿", "jieba").unwrap(),
            MatchExpr::Helper {
                function: "jieba_query",
                query: "矿".to_string(),
            }
        );
    }

    #[test]
    fn distinct_terms_merges_variants_in_first_seen_order() {
        let terms = vec![
            IndexedTerm {
                surface: "知".to_string(),
                variants: vec!["z".to_string()],
            },
            IndexedTerm {
                surface: "识".to_string(),
                variants: vec![],
            },
            IndexedTerm {
                surface: "知".to_string(),
                variants: vec!["zhi".to_string(), "z".to_string()],
            },
        ];
        assert_eq!(
            distinct_terms(&terms),
            vec![
                IndexedTerm {
                    surface: "知".to_string(),
                    variants: vec!["z".to_string(), "zhi".to_string()],
                },
                IndexedTerm {
                    surface: "识".to_string(),
                    variants: vec![],
                },
            ]
        );
    }

    #[test]
    fn native_preview_preserves_colocated_variants() {
        let conn = Connection::open_in_memory().unwrap();
        lode_simple_native::register_connection(&conn, false).unwrap();
        let terms = tokenize_preview(&conn, "知识").unwrap();
        assert_eq!(
            terms,
            vec![
                IndexedTerm {
                    surface: "知".to_string(),
                    variants: vec!["z".to_string(), "zhi".to_string()],
                },
                IndexedTerm {
                    surface: "识".to_string(),
                    variants: vec![
                        "s".to_string(),
                        "shi".to_string(),
                        "z".to_string(),
                        "zhi".to_string(),
                    ],
                },
            ]
        );
    }

    #[test]
    fn builtin_preview_uses_unicode61_tokenizer() {
        let terms = preview_index_terms("unicode61", "Café, Rocks").unwrap();
        assert_eq!(
            terms,
            vec![
                IndexedTerm {
                    surface: "cafe".to_string(),
                    variants: vec![],
                },
                IndexedTerm {
                    surface: "rocks".to_string(),
                    variants: vec![],
                },
            ]
        );
    }

    #[test]
    fn builtin_preview_uses_trigram_tokenizer() {
        let terms = preview_index_terms("trigram", "矿石探矿").unwrap();
        assert_eq!(
            terms,
            vec![
                IndexedTerm {
                    surface: "矿石探".to_string(),
                    variants: vec![],
                },
                IndexedTerm {
                    surface: "石探矿".to_string(),
                    variants: vec![],
                },
            ]
        );
    }

    #[test]
    fn preview_index_terms_uses_the_same_document_stream_for_jieba() {
        let simple = preview_index_terms("simple", "知识").unwrap();
        let jieba = preview_index_terms("jieba", "知识").unwrap();
        assert_eq!(simple, jieba);
        assert!(simple.iter().all(|term| !term.variants.is_empty()));
    }

    #[test]
    fn preview_index_terms_rejects_unknown_strategy() {
        let error = preview_index_terms("unknown", "text").unwrap_err();
        assert!(error.to_string().contains("unknown tokenizer strategy"));
    }
}
