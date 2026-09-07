#![warn(clippy::pedantic)]

//! User-facing message templates.
//!
//! Exception messages are diagnostics; user text comes from this table.
//! Mirrors Python's `messages.py`: each entry carries an `error` sentence and
//! an optional `hint`, with `{placeholder}` substitution done by the caller.

/// Curated user-facing strings for the CLI surface.
pub const STUCK_A_LODE: &str = "Struck a lode";
pub const STUMBLED: &str = "stumbled";
pub const DRY_HOLE: &str = "dry hole";

/// One curated user-facing message: an error sentence plus an optional hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorText {
    /// The primary sentence describing what went wrong.
    pub error: &'static str,
    /// The optional follow-up telling the user what to do about it.
    pub hint: Option<&'static str>,
}

const fn msg(error: &'static str, hint: Option<&'static str>) -> ErrorText {
    ErrorText { error, hint }
}

/// The message table, keyed by stable string keys (matching the Python
/// `messages.py` keys so the two surfaces stay comparable).
pub const MESSAGES: &[(&str, ErrorText)] = &[
    (
        "no_index",
        msg(
            "This workspace has no lode yet at {index_path}.",
            Some("Run `lode mine` to mine it first."),
        ),
    ),
    (
        "invalid_query",
        msg(
            "The lode cannot guess what you seek.",
            Some("Provide a query to start prospecting."),
        ),
    ),
    (
        "invalid_digest",
        msg(
            "This digest cannot identify an ore: {digest}.",
            Some("Make sure the digest is valid."),
        ),
    ),
    (
        "not_found",
        msg(
            "This lode contains no ore with digest {digest}.",
            Some("Make sure both the digest and the workspace are correct."),
        ),
    ),
    (
        "ambiguous",
        msg(
            "There are {count} ores matching {digest}.",
            Some("Use a longer prefix to identify a single ore."),
        ),
    ),
];

/// Look up a message template by key.
///
/// # Panics
///
/// Panics on an unknown key: every key must exist in the table (a missing
/// entry is a programming error, not a runtime condition).
#[must_use]
pub fn require(key: &str) -> ErrorText {
    let (_, text) = MESSAGES
        .iter()
        .find(|(k, _)| *k == key)
        .unwrap_or_else(|| panic!("missing message template: {key}"));
    *text
}

/// Substitute `{placeholder}` slots with values (plain replacement, no
/// escaping needed — the templates are compile-time constants).
#[must_use]
pub fn format(template: &str, values: &[(&str, String)]) -> String {
    let mut out = template.to_string();
    for (slot, value) in values {
        out = out.replace(&format!("{{{slot}}}"), value);
    }
    out
}
