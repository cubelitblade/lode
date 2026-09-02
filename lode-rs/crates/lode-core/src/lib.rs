//! lode core library.
//!
//! The single business-logic crate. Both `lode-cli` and `lode-mcp` are thin
//! adapters over this crate.

pub mod config;
pub mod errors;
pub mod relpath;

pub mod embeddings;
pub mod index;
pub mod ingestion;
pub mod lexical;
pub mod messages;

pub use errors::{Error, Result};
