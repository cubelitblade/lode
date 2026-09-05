#![warn(clippy::pedantic)]

//! Embedder abstraction.
//!
//! Core modules depend only on this interface, never on a concrete
//! implementation, so the model backend (local TEI, hosted API, in-process
//! sentence-transformers) can be swapped via configuration.

/// An embedding provider.
///
/// All implementations:
/// - expose `model_id` and `dimension` (resolved lazily; construction is
///   side-effect free, so creating an embedder never touches the network)
/// - embed a list of documents (bulk) or a single query string
/// - always return L2-normalized vectors, so that cosine similarity ==
///   dot product and consumers can rank by dot
pub trait Embedder {
    /// Model identifier. May trigger a metadata request on first access.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Embedding`] when the underlying provider fails
    /// to advertise a model id.
    fn model_id(&self) -> crate::Result<String>;

    /// Vector dimension. May trigger a probe request on first access.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Embedding`] when probing the provider yields no
    /// usable vector.
    fn dimension(&self) -> crate::Result<usize>;

    /// Embed a batch of documents. Empty input returns an empty list.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Embedding`] when the provider rejects the batch
    /// or responds with fewer vectors than inputs.
    fn embed_documents(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>>;

    /// Embed a single query string.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Embedding`] when the provider fails to return
    /// exactly one vector for the query.
    fn embed_query(&self, text: &str) -> crate::Result<Vec<f32>>;
}

/// L2-normalize each vector, returning new vectors (inputs are untouched).
///
/// Zero vectors are returned as-is (their norm is 0; dividing by it would
/// produce NaN). Re-normalizing an already-normalized vector is a no-op up
/// to floating-point error, so implementations may call this unconditionally
/// as a safety net.
#[must_use]
pub fn l2_normalize(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .map(|vector| {
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                vector.clone()
            } else {
                let scale = 1.0 / norm;
                vector.iter().map(|x| x * scale).collect()
            }
        })
        .collect()
}

/// Convert an `f64` coordinate to `f32`, accepting intentional precision loss.
///
/// Upstream embedding endpoints deliver coordinates as IEEE-754 doubles, but
/// the pipeline stores vectors as singles everywhere downstream. Dropping the
/// low-order mantissa bits is intrinsic to that representation choice, so the
/// trade-off is stated once here rather than repeated at every call site.
#[expect(
    clippy::cast_possible_truncation,
    reason = "doubles from upstream endpoints; single-precision storage is deliberate"
)]
#[must_use]
pub fn embedding_scalar_to_f32(scalar: f64) -> f32 {
    scalar as f32
}

#[cfg(test)]
mod tests {
    use super::l2_normalize;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn normalizes_to_unit_length() {
        let vectors = vec![vec![3.0, 4.0]];
        let out = l2_normalize(&vectors);
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx(norm, 1.0), "expected unit norm, got {norm}");
        assert!(approx(out[0][0], 0.6) && approx(out[0][1], 0.8));
    }

    #[test]
    fn zero_vector_is_returned_as_is() {
        let vectors = vec![vec![0.0, 0.0, 0.0]];
        let out = l2_normalize(&vectors);
        assert_eq!(out[0], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(l2_normalize(&[]).is_empty());
    }

    #[test]
    fn inputs_are_not_mutated() {
        let vectors = vec![vec![3.0, 4.0]];
        let _ = l2_normalize(&vectors);
        assert_eq!(vectors[0], vec![3.0, 4.0]);
    }
}
