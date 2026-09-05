#![warn(clippy::pedantic)]

//! Helper constructors for embedding-related errors.
//!
//! lode-core funnels all embedding failures through [`crate::Error::Embedding`];
//! this module centralizes the wording so backends raise consistently.

use crate::Error;

/// Wrap a generic message as an embedding error.
pub fn err(message: impl Into<String>) -> Error {
    Error::Embedding(message.into())
}

/// Report that a required piece of metadata came back empty.
pub fn missing_field(kind: &str, detail: impl Into<String>) -> Error {
    err(format!("missing {kind}: {}", detail.into()))
}
