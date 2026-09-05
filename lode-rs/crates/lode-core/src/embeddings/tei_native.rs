#![warn(clippy::pedantic)]

//! Hugging Face Text Embeddings Inference (TEI) native API.
//!
//! Endpoints:
//! - `GET {base}/info`  -> discovers the served model id
//! - `POST {base}/embed` -> embeds batches
//!
//! Returns bare vectors in input order (no `index` wrapper). `424` (backend
//! inference failure) is not retryable; only `429` is.

use super::{
    errors,
    http::{HttpBackend, HttpEmbedder},
};

const PATH_INFO: &str = "/info";
const PATH_EMBED: &str = "/embed";

/// Only rate-limit responses are retried; `424` means the model failed to
/// load and retrying will not help.
const RETRYABLE_STATUS: &[u16] = &[429];

/// Builder producing a TEI-native `HttpEmbedder`.
pub struct TeiNativeBuilder<'a> {
    base_url: &'a str,
    model: Option<String>,
    dimension: Option<usize>,
    output_dimension: Option<usize>,
    batch_size: usize,
    retries: u32,
    api_key: Option<String>,
    truncate: bool,
    truncation_direction: String,
    client: Box<dyn super::http::HttpClient>,
}

impl<'a> TeiNativeBuilder<'a> {
    #[must_use]
    pub fn new(base_url: &'a str, client: Box<dyn super::http::HttpClient>) -> Self {
        Self {
            base_url,
            model: None,
            dimension: None,
            output_dimension: None,
            batch_size: super::http::DEFAULT_BATCH_SIZE,
            retries: 3,
            api_key: None,
            truncate: false,
            truncation_direction: "right".to_string(),
            client,
        }
    }

    #[must_use]
    pub fn model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }

    #[must_use]
    pub fn dimension(mut self, dimension: Option<usize>) -> Self {
        self.dimension = dimension;
        self
    }

    #[must_use]
    pub fn output_dimension(mut self, d: Option<usize>) -> Self {
        self.output_dimension = d;
        self
    }

    #[must_use]
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    #[must_use]
    pub fn retries(mut self, r: u32) -> Self {
        self.retries = r;
        self
    }

    #[must_use]
    pub fn api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }

    #[must_use]
    pub fn truncate(mut self, yes: bool) -> Self {
        self.truncate = yes;
        self
    }

    #[must_use]
    pub fn truncation_direction(mut self, dir: String) -> Self {
        self.truncation_direction = dir;
        self
    }

    #[must_use]
    pub fn build(self) -> HttpEmbedder {
        HttpEmbedder::new(
            Box::new(TeiNativeBackend {
                truncate: self.truncate,
                truncation_direction: self.truncation_direction,
            }),
            self.client,
            self.base_url,
            self.model,
            self.dimension,
            self.output_dimension,
            self.batch_size,
            self.retries,
            self.api_key,
        )
    }
}

struct TeiNativeBackend {
    truncate: bool,
    truncation_direction: String,
}

impl HttpBackend for TeiNativeBackend {
    fn backend_label(&self) -> &'static str {
        "TEI-native"
    }

    fn retryable_status(&self) -> &'static [u16] {
        RETRYABLE_STATUS
    }

    fn fetch_model_id(&self, http: &HttpEmbedder) -> crate::Result<String> {
        let url = format!("{}{PATH_INFO}", http.base_url());
        let resp = http.request("GET", &url, None)?;
        let parsed: serde_json::Value = serde_json::from_str(&resp.body)
            .map_err(|e| errors::err(format!("invalid JSON from {url}: {e}")))?;
        let id = parsed
            .get("model_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| errors::missing_field("model_id", url))?;
        Ok(id.to_string())
    }

    fn embed(&self, http: &HttpEmbedder, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        let url = format!("{}{PATH_EMBED}", http.base_url());

        let inputs: Vec<serde_json::Value> = texts.iter().map(|t| t.clone().into()).collect();
        let mut map = serde_json::Map::new();
        map.insert("inputs".into(), serde_json::Value::Array(inputs));
        // L2 normalization is always requested server-side; the client also
        // normalizes (idempotent) to keep the Embedder contract centralized.
        map.insert("normalize".into(), true.into());
        map.insert("truncate".into(), self.truncate.into());
        map.insert(
            "truncation_direction".into(),
            self.truncation_direction.clone().into(),
        );
        if let Some(out_dim) = http.output_dimension() {
            map.insert("dimensions".into(), out_dim.into());
        }

        let payload = serde_json::Value::Object(map).to_string();
        let resp = http.request("POST", &url, Some(&payload))?;
        let parsed: serde_json::Value = serde_json::from_str(&resp.body)
            .map_err(|e| errors::err(format!("invalid JSON from {url}: {e}")))?;

        let arr = parsed
            .as_array()
            .ok_or_else(|| errors::missing_field("top-level array", &url))?;
        let vectors = arr
            .iter()
            .map(|v| {
                let inner = v
                    .as_array()
                    .ok_or_else(|| errors::missing_field("vector", &url))?;
                inner
                    .iter()
                    .map(|x| x.as_f64().map(crate::embeddings::embedding_scalar_to_f32))
                    .collect::<Option<Vec<f32>>>()
                    .ok_or_else(|| errors::err(format!("non-numeric embedding value from {url}")))
            })
            .collect::<crate::Result<Vec<Vec<f32>>>>()?;
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::base::Embedder;
    use crate::embeddings::testing::{MockHttpClientHandle, MockResponse, shared_mock};

    const MODEL_ID: &str = "BAAI/bge-m3";
    const DIM: usize = 3;

    fn info_response() -> MockResponse {
        MockResponse::json(200, &format!(r#"{{"model_id":"{MODEL_ID}"}}"#))
    }

    fn embed_response(texts: usize, dim: usize) -> MockResponse {
        let rows: Vec<serde_json::Value> = (0..texts)
            .map(|_| serde_json::Value::Array(vec![serde_json::Value::from(1.0); dim]))
            .collect();
        MockResponse::json(200, &serde_json::Value::Array(rows).to_string())
    }

    fn make_embedder(client: MockHttpClientHandle) -> HttpEmbedder {
        TeiNativeBuilder::new("http://localhost:8080", Box::new(client))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}))
    }

    #[test]
    fn discovers_model_id_from_info_endpoint() {
        let client = shared_mock(vec![info_response()]);
        let emb = TeiNativeBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .build()
            .with_backoff(Box::new(|_| {}));
        assert_eq!(emb.model_id().unwrap(), MODEL_ID);
        let reqs = client.borrow().requests();
        assert_eq!(reqs[0].method, "GET");
        assert_eq!(reqs[0].url, "http://localhost:8080/info");
    }

    #[test]
    fn embed_payload_contains_inputs_and_truncate() {
        let client = shared_mock(vec![embed_response(2, DIM)]);
        let emb = TeiNativeBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .dimension(Some(DIM))
            .truncate(true)
            .truncation_direction("left".to_string())
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["a".to_string(), "b".to_string()])
            .unwrap();
        let reqs = client.borrow().requests();
        assert_eq!(reqs[0].url, "http://localhost:8080/embed");
        let body: serde_json::Value =
            serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["inputs"], serde_json::json!(["a", "b"]));
        assert_eq!(body["truncate"], serde_json::json!(true));
        assert_eq!(body["truncation_direction"], serde_json::json!("left"));
    }

    #[test]
    fn normalize_true_is_sent() {
        let client = shared_mock(vec![embed_response(1, DIM)]);
        let emb = TeiNativeBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["x".to_string()]).unwrap();
        let reqs = client.borrow().requests();
        let body: serde_json::Value =
            serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["normalize"], serde_json::json!(true));
    }

    #[test]
    fn output_dimension_is_sent_when_configured() {
        let client = shared_mock(vec![embed_response(1, DIM)]);
        let emb = TeiNativeBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .dimension(Some(DIM))
            .output_dimension(Some(2))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["x".to_string()]).unwrap();
        let reqs = client.borrow().requests();
        let body: serde_json::Value =
            serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["dimensions"], serde_json::json!(2));
    }

    #[test]
    fn parses_bare_vectors_in_input_order() {
        let client = shared_mock(vec![embed_response(2, DIM)]);
        let emb = TeiNativeBuilder::new("http://localhost:8080", Box::new(client))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}));
        let vecs = emb
            .embed_documents(&["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(vecs.len(), 2);
        // [1.0; DIM] L2-normalizes to 1/sqrt(DIM) per component.
        #[expect(
            clippy::cast_precision_loss,
            reason = "fixture computes sqrt denominator from a tiny constant dimension"
        )]
        let expected = 1.0 / (DIM as f32).sqrt();
        assert!(vecs[0].iter().all(|x| (x - expected).abs() < 1e-5));
    }

    #[test]
    fn status_424_is_not_retried() {
        let client = shared_mock(vec![MockResponse::json(424, "model failed to load")]);
        let emb = make_embedder(client.clone());
        let err = emb.embed_documents(&["x".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("HTTP 424"),
            "unexpected error: {err}"
        );
        assert_eq!(client.borrow().request_count(), 1);
    }

    #[test]
    fn status_429_is_retried() {
        let client = shared_mock(vec![
            MockResponse::json(429, "rate limited"),
            embed_response(1, DIM),
        ]);
        let emb = make_embedder(client.clone());
        let vecs = emb.embed_documents(&["x".to_string()]).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(client.borrow().request_count(), 2);
    }

    #[test]
    fn embed_errors_on_non_numeric_value() {
        let client = shared_mock(vec![MockResponse::json(200, r#"[["oops"]]"#)]);
        let emb = TeiNativeBuilder::new("http://localhost:8080", Box::new(client))
            .dimension(Some(1))
            .build()
            .with_backoff(Box::new(|_| {}));
        let err = emb.embed_documents(&["x".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("non-numeric"),
            "unexpected error: {err}"
        );
    }
}
