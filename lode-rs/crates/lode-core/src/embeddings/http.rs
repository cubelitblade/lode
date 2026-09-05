#![warn(clippy::pedantic)]

//! Shared HTTP plumbing for embedding clients that speak a REST API.
//!
//! Owns everything concrete backends need: lazy metadata resolution
//! (`model_id` / `dimension`), batched embedding, retry with exponential
//! backoff, and API-key headers. Backends only describe what differs:
//! the backend label, retryable statuses, model discovery, and payload
//! construction / response parsing.

use std::cell::RefCell;

use super::base::{Embedder, l2_normalize};

/// TEI advertises `max_client_batch_size=32`; other servers accept more, but
/// staying at or below it keeps the client safe across backends.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Used to auto-detect the vector dimension at startup.
pub const PROBE_TEXT: &str = "ping";

// ---------------------------------------------------------------------------
// Transport abstraction
// ---------------------------------------------------------------------------

/// A minimal HTTP response.
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// A transport-level error (unreachable endpoint, malformed response).
#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    #[error("transport error: {0}")]
    Transport(String),
}

/// Abstraction over the HTTP client so tests can inject a mock.
pub trait HttpClient {
    /// Perform a synchronous request.
    ///
    /// # Errors
    ///
    /// Surfaces transport failures as [`HttpError`]; the caller decides
    /// whether they are worth retrying.
    fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
        headers: &[(String, String)],
    ) -> std::result::Result<HttpResponse, HttpError>;
}

/// Production `HttpClient` backed by `reqwest::blocking`.
pub struct ReqwestHttpClient {
    client: reqwest::blocking::Client,
}

impl ReqwestHttpClient {
    /// Construct a client with the given overall timeout budget.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Embedding`] when `reqwest` declines to build a
    /// client (usually an unusable TLS/system configuration).
    pub fn new(timeout_secs: f64) -> crate::Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout_secs))
            .build()
            .map_err(|e| crate::Error::Embedding(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { client })
    }
}

impl HttpClient for ReqwestHttpClient {
    fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
        headers: &[(String, String)],
    ) -> std::result::Result<HttpResponse, HttpError> {
        let method = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|e| HttpError::Transport(e.to_string()))?;
        let mut req = self.client.request(method, url);
        for (key, value) in headers {
            req = req.header(key.as_str(), value.as_str());
        }
        if let Some(payload) = body {
            req = req
                .header("Content-Type", "application/json")
                .body(payload.to_string());
        }
        let resp = req
            .send()
            .map_err(|e| HttpError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .map_err(|e| HttpError::Transport(e.to_string()))?;
        Ok(HttpResponse { status, body: text })
    }
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Backend-specific behaviour for an HTTP embedding API.
///
/// Implementors only define what differs between providers.  All shared
/// logic (retry, batching, lazy metadata) lives in `HttpEmbedder`.
pub trait HttpBackend {
    /// Backend tag used in log messages (e.g. `"OpenAI-compatible"`).
    fn backend_label(&self) -> &'static str;

    /// HTTP status codes safe to retry; backends narrow this per API.
    fn retryable_status(&self) -> &'static [u16];

    /// Discover the model id from the endpoint (`GET /v1/models`, etc.).
    ///
    /// # Errors
    ///
    /// Delegates to [`HttpEmbedder::request`] and additionally validates the
    /// discovered model-id field.
    fn fetch_model_id(&self, http: &HttpEmbedder) -> crate::Result<String>;

    /// Build the payload, call the endpoint, and parse the response for one batch.
    ///
    /// # Errors
    ///
    /// Propagates request failures and validation errors (malformed bodies,
    /// non-numeric components, shape mismatch).
    fn embed(&self, http: &HttpEmbedder, texts: &[String]) -> crate::Result<Vec<Vec<f32>>>;
}

// ---------------------------------------------------------------------------
// HttpEmbedder
// ---------------------------------------------------------------------------

/// Shared HTTP embedding client.
///
/// Construction is side-effect free: metadata (model id, dimension) is
/// resolved lazily on first access, so instantiating an embedder never
/// touches the network.
pub struct HttpEmbedder {
    backend: Box<dyn HttpBackend>,
    base_url: String,
    batch_size: usize,
    retries: u32,
    api_key: Option<String>,
    client: Box<dyn HttpClient>,
    backoff: Box<dyn Fn(u32)>,

    // Config overrides (None = resolve lazily).
    model: Option<String>,
    dimension: Option<usize>,
    output_dimension: Option<usize>,

    // Lazy-resolved caches.
    fetched_model: RefCell<Option<String>>,
    fetched_dimension: RefCell<Option<usize>>,
}

impl HttpEmbedder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Box<dyn HttpBackend>,
        client: Box<dyn HttpClient>,
        base_url: &str,
        model: Option<String>,
        dimension: Option<usize>,
        output_dimension: Option<usize>,
        batch_size: usize,
        retries: u32,
        api_key: Option<String>,
    ) -> Self {
        Self {
            backend,
            base_url: base_url.trim_end_matches('/').to_string(),
            batch_size: batch_size.max(1),
            retries,
            api_key,
            client,
            backoff: Box::new(exponential_backoff),
            model,
            dimension,
            output_dimension,
            fetched_model: RefCell::new(None),
            fetched_dimension: RefCell::new(None),
        }
    }

    /// Override the retry backoff (tests inject a no-op to avoid sleeping).
    #[must_use]
    pub fn with_backoff(mut self, f: Box<dyn Fn(u32)>) -> Self {
        self.backoff = f;
        self
    }

    /// The endpoint base URL (trailing slash stripped).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Optional output dimension override (payload `dimensions` field).
    pub fn output_dimension(&self) -> Option<usize> {
        self.output_dimension
    }

    /// Issue a JSON request with retry and exponential backoff.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Embedding`] when the endpoint stays unhealthy
    /// past the retry budget, or replies with a non-retryable HTTP status.
    pub fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
    ) -> crate::Result<HttpResponse> {
        let mut headers: Vec<(String, String)> = Vec::new();
        if let Some(key) = &self.api_key {
            headers.push(("Authorization".into(), format!("Bearer {key}")));
        }

        let retryable = self.backend.retryable_status();
        let mut attempt = 0_u32;
        loop {
            match self.client.request(method, url, body, &headers) {
                Ok(resp) => {
                    if resp.status < 400
                        || !retryable.contains(&resp.status)
                        || attempt >= self.retries
                    {
                        if resp.status >= 400 {
                            return Err(crate::Error::Embedding(format!(
                                "embedding endpoint {url} returned HTTP {}: {}",
                                resp.status, resp.body,
                            )));
                        }
                        return Ok(resp);
                    }
                    attempt += 1;
                    (self.backoff)(attempt);
                }
                Err(exc) => {
                    if attempt >= self.retries {
                        return Err(crate::Error::Embedding(format!(
                            "embedding endpoint {url} is unreachable after {} attempts: {exc}",
                            attempt + 1,
                        )));
                    }
                    attempt += 1;
                    (self.backoff)(attempt);
                }
            }
        }
    }

    /// Optional API key header.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

impl Embedder for HttpEmbedder {
    fn model_id(&self) -> crate::Result<String> {
        // Config override wins.
        if let Some(m) = &self.model {
            return Ok(m.clone());
        }
        // Lazy: fetch once, then cache.
        let mut fetched = self.fetched_model.borrow_mut();
        if fetched.is_none() {
            *fetched = Some(self.backend.fetch_model_id(self)?);
        }
        Ok(fetched.clone().expect("just set"))
    }

    fn dimension(&self) -> crate::Result<usize> {
        if let Some(d) = self.dimension {
            return Ok(d);
        }
        let mut fetched = self.fetched_dimension.borrow_mut();
        if fetched.is_none() {
            let probe = self.backend.embed(self, &[PROBE_TEXT.to_string()])?;
            let dim = probe
                .first()
                .ok_or_else(|| {
                    crate::Error::Embedding("dimension probe returned no vectors".into())
                })?
                .len();
            *fetched = Some(dim);
        }
        Ok(fetched.expect("just set"))
    }

    fn embed_documents(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            let mut batch_vectors = self.backend.embed(self, batch)?;
            if batch_vectors.len() != batch.len() {
                return Err(crate::Error::Embedding(format!(
                    "embedding endpoint returned {} vectors for {} inputs",
                    batch_vectors.len(),
                    batch.len(),
                )));
            }
            vectors.append(&mut batch_vectors);
        }
        // L2 normalization is always applied client-side (cosine == dot
        // contract); the option is not exposed to users.
        Ok(l2_normalize(&vectors))
    }

    fn embed_query(&self, text: &str) -> crate::Result<Vec<f32>> {
        let vectors = self.backend.embed(self, &[text.to_string()])?;
        if vectors.len() != 1 {
            return Err(crate::Error::Embedding(format!(
                "embedding endpoint returned {} vectors for 1 input",
                vectors.len(),
            )));
        }
        let mut normalized = l2_normalize(&vectors);
        Ok(normalized.swap_remove(0))
    }
}

/// Exponential backoff: 2s, 4s, 8s, 8s, …  (matches Python `_backoff`).
fn exponential_backoff(attempt: u32) {
    let secs = (2_u32 << (attempt - 1)).min(8);
    std::thread::sleep(std::time::Duration::from_secs(secs.into()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::testing::{MockHttpClientHandle, MockResponse, shared_mock};

    /// A minimal backend that echoes a fixed vector per input.
    struct EchoBackend;

    impl HttpBackend for EchoBackend {
        fn backend_label(&self) -> &'static str {
            "echo"
        }

        fn retryable_status(&self) -> &'static [u16] {
            &[429]
        }

        fn fetch_model_id(&self, http: &HttpEmbedder) -> crate::Result<String> {
            let resp = http.request("GET", &format!("{}/info", http.base_url()), None)?;
            let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
            Ok(v["model_id"].as_str().unwrap().to_string())
        }

        fn embed(&self, http: &HttpEmbedder, _texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
            let resp = http.request("POST", &format!("{}/embed", http.base_url()), None)?;
            let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
            let arr = v.as_array().unwrap();
            Ok(arr
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| crate::embeddings::embedding_scalar_to_f32(x.as_f64().unwrap()))
                        .collect()
                })
                .collect())
        }
    }

    fn embedder(client: MockHttpClientHandle, batch_size: usize) -> HttpEmbedder {
        HttpEmbedder::new(
            Box::new(EchoBackend),
            Box::new(client),
            "http://localhost:8080/",
            None,
            None,
            None,
            batch_size,
            3,
            None,
        )
        .with_backoff(Box::new(|_| {}))
    }

    fn vec_response(texts: usize, dim: usize) -> MockResponse {
        let rows: Vec<serde_json::Value> = (0..texts)
            .map(|_| serde_json::Value::Array(vec![serde_json::Value::from(1.0); dim]))
            .collect();
        MockResponse::json(200, &serde_json::Value::Array(rows).to_string())
    }

    #[test]
    fn construction_is_side_effect_free() {
        // No scripted responses: any request would panic the mock.
        let emb = embedder(shared_mock(vec![]), 2);
        assert_eq!(emb.base_url(), "http://localhost:8080");
    }

    #[test]
    fn model_id_is_lazy_and_cached() {
        let client = shared_mock(vec![MockResponse::json(
            200,
            r#"{"model_id":"test-model"}"#,
        )]);
        let emb = embedder(client.clone(), 2);
        assert_eq!(emb.model_id().unwrap(), "test-model");
        assert_eq!(emb.model_id().unwrap(), "test-model");
        assert_eq!(
            client.borrow().request_count(),
            1,
            "second access must not re-fetch"
        );
    }

    #[test]
    fn explicit_model_skips_request() {
        let client = shared_mock(vec![]);
        let emb = HttpEmbedder::new(
            Box::new(EchoBackend),
            Box::new(client.clone()),
            "http://localhost:8080",
            Some("cfg-model".to_string()),
            None,
            None,
            2,
            3,
            None,
        );
        assert_eq!(emb.model_id().unwrap(), "cfg-model");
        assert_eq!(client.borrow().request_count(), 0);
    }

    #[test]
    fn dimension_is_lazy_and_probes() {
        let client = shared_mock(vec![vec_response(1, 4)]);
        let emb = embedder(client.clone(), 2);
        assert_eq!(emb.dimension().unwrap(), 4);
        assert_eq!(emb.dimension().unwrap(), 4);
        assert_eq!(
            client.borrow().request_count(),
            1,
            "second access must not re-probe"
        );
    }

    #[test]
    fn embed_documents_batches_by_batch_size() {
        let client = shared_mock(vec![
            vec_response(2, 3),
            vec_response(2, 3),
            vec_response(1, 3),
        ]);
        let emb = embedder(client.clone(), 2);
        let texts: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        let vecs = emb.embed_documents(&texts).unwrap();
        assert_eq!(vecs.len(), 5);
        assert_eq!(client.borrow().request_count(), 3);
    }

    #[test]
    fn empty_input_makes_no_request() {
        let client = shared_mock(vec![]);
        let emb = embedder(client.clone(), 2);
        assert!(emb.embed_documents(&[]).unwrap().is_empty());
        assert_eq!(client.borrow().request_count(), 0);
    }

    #[test]
    fn normalization_applied_by_default() {
        let client = shared_mock(vec![vec_response(1, 4)]);
        let emb = embedder(client, 2);
        let vecs = emb.embed_documents(&["x".to_string()]).unwrap();
        let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn retries_then_succeeds() {
        let client = shared_mock(vec![
            MockResponse::json(429, "rate limited"),
            vec_response(1, 2),
        ]);
        let emb = embedder(client.clone(), 2);
        let vecs = emb.embed_documents(&["x".to_string()]).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(client.borrow().request_count(), 2);
    }

    #[test]
    fn gives_up_after_max_retries() {
        let client = shared_mock(vec![
            MockResponse::json(429, "rate limited"),
            MockResponse::json(429, "rate limited"),
            MockResponse::json(429, "rate limited"),
            MockResponse::json(429, "rate limited"),
        ]);
        let emb = embedder(client.clone(), 2);
        let err = emb.embed_documents(&["x".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("HTTP 429"),
            "unexpected error: {err}"
        );
        assert_eq!(client.borrow().request_count(), 4);
    }

    #[test]
    fn non_retryable_status_errors_immediately() {
        let client = shared_mock(vec![MockResponse::json(400, "bad request")]);
        let emb = embedder(client.clone(), 2);
        let err = emb.embed_documents(&["x".to_string()]).unwrap_err();
        assert!(err.to_string().contains("HTTP 400"));
        assert_eq!(client.borrow().request_count(), 1);
    }

    #[test]
    fn api_key_sends_bearer_header() {
        let client = shared_mock(vec![vec_response(1, 2)]);
        let emb = HttpEmbedder::new(
            Box::new(EchoBackend),
            Box::new(client.clone()),
            "http://localhost:8080",
            None,
            Some(2),
            None,
            2,
            3,
            Some("secret".to_string()),
        )
        .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["x".to_string()]).unwrap();
        let reqs = client.borrow().requests();
        assert_eq!(
            reqs[0].headers[0],
            ("Authorization".to_string(), "Bearer secret".to_string())
        );
    }

    #[test]
    fn embed_query_returns_single_vector() {
        let client = shared_mock(vec![vec_response(1, 3)]);
        let emb = embedder(client, 2);
        let vec = emb.embed_query("x").unwrap();
        assert_eq!(vec.len(), 3);
    }

    #[test]
    fn embed_query_errors_on_empty_response() {
        let client = shared_mock(vec![MockResponse::json(200, "[]")]);
        let emb = embedder(client, 2);
        let err = emb.embed_query("x").unwrap_err();
        assert!(
            err.to_string().contains("0 vectors for 1 input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn embed_documents_errors_on_vector_count_mismatch() {
        let client = shared_mock(vec![vec_response(1, 3)]);
        let emb = embedder(client, 2);
        let err = emb
            .embed_documents(&["a".to_string(), "b".to_string()])
            .unwrap_err();
        assert!(
            err.to_string().contains("1 vectors for 2 inputs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dimension_errors_on_empty_probe() {
        let client = shared_mock(vec![MockResponse::json(200, "[]")]);
        let emb = embedder(client, 2);
        let err = emb.dimension().unwrap_err();
        assert!(
            err.to_string().contains("probe returned no vectors"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn trailing_slash_is_stripped() {
        let emb = embedder(shared_mock(vec![]), 2);
        assert_eq!(emb.base_url(), "http://localhost:8080");
    }
}
