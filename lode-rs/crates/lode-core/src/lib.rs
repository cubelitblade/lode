//! lode core library.
//!
//! The single business-logic crate. Both `lode-cli` and `lode-mcp` are thin
//! adapters over this crate.

pub mod config;
pub mod errors;
pub mod relpath;

pub mod embeddings;
pub mod fts;
pub mod index;
pub mod ingestion;
pub mod messages;

pub use errors::{Error, Result};
