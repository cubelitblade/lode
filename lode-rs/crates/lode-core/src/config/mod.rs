//! Layered configuration.
//!
//! Precedence: `defaults < TOML < environment variables < constructor kwargs`.
//! The TOML source is treated as replaceable for future formats.

pub mod layered;

use serde::{Deserialize, Serialize};

/// Top-level application settings model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    /// Output / ignore switch configuration.
    #[serde(default)]
    pub app: AppConfig,
    /// Embedding provider configuration.
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    /// Retrieval configuration.
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    /// Chunking configuration.
    #[serde(default)]
    pub chunking: ChunkingConfig,
    /// FTS configuration.
    #[serde(default)]
    pub fts: FtsConfig,
    /// Ignore rules.
    #[serde(default)]
    pub ignore: IgnoreConfig,
}

/// App-level configuration (output, ignore).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {}

/// Embedding provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingConfig {}

/// Retrieval configuration (norm, fusion).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalConfig {}

/// Chunking configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkingConfig {}

/// FTS configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FtsConfig {}

/// Ignore rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IgnoreConfig {}
