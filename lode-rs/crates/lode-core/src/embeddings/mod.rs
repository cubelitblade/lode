#![warn(clippy::pedantic)]

//! Embedding providers.

pub mod base;
pub mod errors;
pub mod http;
pub mod ollama;
pub mod openai_compatible;
pub mod tei_native;

#[cfg(test)]
pub mod testing;

pub use base::{Embedder, embedding_scalar_to_f32, l2_normalize};
