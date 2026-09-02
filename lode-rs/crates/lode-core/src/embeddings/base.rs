//! Embedder abstraction.

/// An embedding provider.
pub trait Embedder {
    /// Identifier of the underlying model.
    fn model_id(&self) -> &str;

    /// Embedding dimension.
    fn dimension(&self) -> usize;

    /// Embed a batch of documents.
    fn embed_documents(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>>;

    /// Embed a single query string.
    fn embed_query(&self, text: &str) -> crate::Result<Vec<f32>>;
}
