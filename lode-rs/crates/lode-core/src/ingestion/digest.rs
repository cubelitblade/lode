//! Content addressing primitives for chunk deduplication.
//!
//! A chunk's identity is derived from its normalized text, so unchanged content
//! keeps the same [`chunk_digest`] wherever it appears. The normalization rule
//! is a stability contract: changing it invalidates every stored digest, so it
//! is locked by tests.
//!
//! This module is ported from `src/lode/ingestion/digest.py` and must produce
//! identical output for the same input.

/// Algorithm tag embedded in every digest, so future algorithm switches
/// can be detected without a schema migration.
pub const DIGEST_PREFIX: &str = "blake3:";

/// Collapse runs of horizontal whitespace inside a line. Newlines are kept:
/// chunk boundaries and paragraph structure must survive normalization.
fn normalize_inline_ws(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = false;

    for ch in text.chars() {
        match ch {
            ' ' | '\t' | '\x0c' | '\x0b' => {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            }
            _ => {
                result.push(ch);
                prev_was_space = false;
            }
        }
    }

    result
}

/// Stable canonical form of chunk text.
///
/// Rules (do not change without a full reindex):
/// - strip leading/trailing whitespace (including newlines)
/// - collapse runs of spaces/tabs/form-feed/vertical-tab to a single space
/// - keep internal newlines, paragraph breaks, and case intact
pub fn normalize(text: &str) -> String {
    normalize_inline_ws(text).trim().to_string()
}

/// Content address: `blake3:<hex>` of the normalized text.
///
/// Used for chunk-level identity; identical normalized text always produces
/// the same digest, regardless of which file it came from.
pub fn chunk_digest(text: &str) -> String {
    let normalized = normalize(text);
    let hash = blake3::hash(normalized.as_bytes());
    format!("{DIGEST_PREFIX}{}", hash.to_hex())
}

/// File-level content address: `blake3:<hex>` of the raw bytes.
///
/// Used for rename detection and change attribution; unlike
/// [`chunk_digest`] it hashes the raw bytes, not a normalized form.
pub fn file_digest(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    format!("{DIGEST_PREFIX}{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_whitespace() {
        assert_eq!(normalize("  hello  "), "hello");
        assert_eq!(normalize("\nhello\n"), "hello");
        assert_eq!(normalize("  hello  world  "), "hello world");
    }

    #[test]
    fn normalize_collapses_inline_whitespace() {
        assert_eq!(normalize("hello  world"), "hello world");
        assert_eq!(normalize("hello\t\tworld"), "hello world");
        assert_eq!(normalize("hello \t world"), "hello world");
    }

    #[test]
    fn normalize_keeps_newlines() {
        assert_eq!(normalize("hello\n\nworld"), "hello\n\nworld");
        assert_eq!(normalize("line1\nline2"), "line1\nline2");
    }

    #[test]
    fn chunk_digest_prefix() {
        let digest = chunk_digest("hello");
        assert!(digest.starts_with(DIGEST_PREFIX));
    }

    #[test]
    fn chunk_digest_deterministic() {
        assert_eq!(chunk_digest("hello"), chunk_digest("hello"));
    }

    #[test]
    fn chunk_digest_same_normalized_text() {
        // Leading/trailing whitespace should not affect the digest.
        assert_eq!(chunk_digest("  hello  "), chunk_digest("hello"));
    }

    #[test]
    fn chunk_digest_differs_on_different_text() {
        assert_ne!(chunk_digest("hello"), chunk_digest("world"));
    }

    #[test]
    fn file_digest_prefix() {
        let digest = file_digest(b"hello");
        assert!(digest.starts_with(DIGEST_PREFIX));
    }

    #[test]
    fn file_digest_deterministic() {
        assert_eq!(file_digest(b"hello"), file_digest(b"hello"));
    }

    #[test]
    fn file_digest_matches_chunk_digest_for_clean_text() {
        // When text has no extra whitespace, normalize is a no-op,
        // so file_digest of the raw bytes equals chunk_digest of the text.
        let text = "hello world";
        assert_eq!(file_digest(text.as_bytes()), chunk_digest(text));
    }

    #[test]
    fn normalize_tabs_and_form_feed() {
        // Form-feed (0x0C) and vertical-tab (0x0B) should collapse.
        assert_eq!(normalize("a\x0cb"), "a b");
        assert_eq!(normalize("a\x0bb"), "a b");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   "), "");
    }
}
