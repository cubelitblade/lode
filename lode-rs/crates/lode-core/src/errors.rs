//! Unified error type for lode-core.

/// Result alias for lode-core operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error variant enum for lode-core.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O and filesystem errors.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration load errors.
    #[error("config error: {0}")]
    Config(String),

    /// Errors from the SQLite store.
    #[error("store error: {0}")]
    Store(String),

    /// A chunk vector's dimension differs from the index's vec0 schema.
    ///
    /// Mirrors Python's `DimensionMismatchError`: carries both widths so the
    /// CLI can present a friendly recovery message.
    #[error("dimension mismatch: stored {stored}, got {current}")]
    DimensionMismatch { stored: u32, current: u32 },

    /// Errors from embedding providers.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Errors during ingestion (discovery / extraction / pipeline).
    #[error("ingestion error: {0}")]
    Ingestion(String),

    /// Errors during lexical tokenization.
    #[error("lexical error: {0}")]
    Lexical(String),
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Store(e.to_string())
    }
}
