#![warn(clippy::pedantic)]

//! Ollama native embedding API.
//!
//! Endpoints:
//! - `GET {base}/api/tags`   -> discovers the served model id
//! - `POST {base}/api/embed` -> embeds batches
//!
//! Vectors arrive under `embeddings` in input order. Truncation is always
//! enabled (Ollama truncates long inputs by default).

use super::{
    base::Embedder,
    errors,
    http::{HttpBackend, HttpEmbedder},
};

const PATH_TAGS: &str = "/api/tags";
const PATH_EMBED: &str = "/api/embed";

/// Ollama returns 500/502/503/504 for transient failures; 404 means the
/// model is missing and retrying will not help.
const RETRYABLE_STATUS: &[u16] = &[500, 502, 503, 504];

/// Builder producing an Ollama `HttpEmbedder`.
pub struct OllamaBuilder<'a> {
    base_url: &'a str,
    model: Option<String>,
    dimension: Option<usize>,
    output_dimension: Option<usize>,
    batch_size: usize,
    retries: u32,
    api_key: Option<String>,
    truncate: bool,
    client: Box<dyn super::http::HttpClient>,
}

impl<'a> OllamaBuilder<'a> {
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
            truncate: true,
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
    pub fn build(self) -> HttpEmbedder {
        HttpEmbedder::new(
            Box::new(OllamaBackend {
                truncate: self.truncate,
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

struct OllamaBackend {
    truncate: bool,
}

impl HttpBackend for OllamaBackend {
    fn backend_label(&self) -> &'static str {
        "Ollama"
    }

    fn retryable_status(&self) -> &'static [u16] {
        RETRYABLE_STATUS
    }

    fn fetch_model_id(&self, http: &HttpEmbedder) -> crate::Result<String> {
        let url = format!("{}{PATH_TAGS}", http.base_url());
        let resp = http.request("GET", &url, None)?;
        let parsed: serde_json::Value = serde_json::from_str(&resp.body)
            .map_err(|e| errors::err(format!("invalid JSON from {url}: {e}")))?;
        let arr = parsed
            .get("models")
            .and_then(|v| v.as_array())
            .filter(|arr| !arr.is_empty())
            .ok_or_else(|| errors::missing_field("non-empty models[]", &url))?;
        let id = arr[0]
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| errors::missing_field("model name", &url))?;
        Ok(id.to_string())
    }

    fn embed(&self, http: &HttpEmbedder, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        let url = format!("{}{PATH_EMBED}", http.base_url());

        let mut map = serde_json::Map::new();
        // The Ollama API requires `model`; resolve it lazily when not
        // configured (matches Python, which always sends it).
        map.insert("model".into(), http.model_id()?.into());
        let inputs: Vec<serde_json::Value> = texts.iter().map(|t| t.clone().into()).collect();
        map.insert("input".into(), serde_json::Value::Array(inputs));
        map.insert("truncate".into(), self.truncate.into());
        if let Some(out_dim) = http.output_dimension() {
            map.insert("dimensions".into(), out_dim.into());
        }

        let payload = serde_json::Value::Object(map).to_string();
        let resp = http.request("POST", &url, Some(&payload))?;
        let parsed: serde_json::Value = serde_json::from_str(&resp.body)
            .map_err(|e| errors::err(format!("invalid JSON from {url}: {e}")))?;

        let arr = parsed
            .get("embeddings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| errors::missing_field("embeddings[]", &url))?;
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

    const MODEL_ID: &str = "nomic-embed-text";
    const DIM: usize = 3;

    fn tags_response() -> MockResponse {
        MockResponse::json(200, &format!(r#"{{"models":[{{"name":"{MODEL_ID}"}}]}}"#))
    }

    fn embed_response(texts: usize, dim: usize) -> MockResponse {
        let rows: Vec<serde_json::Value> = (0..texts)
            .map(|_| serde_json::Value::Array(vec![serde_json::Value::from(1.0); dim]))
            .collect();
        MockResponse::json(200, &serde_json::json!({ "embeddings": rows }).to_string())
    }

    fn make_embedder(client: MockHttpClientHandle) -> HttpEmbedder {
        OllamaBuilder::new("http://localhost:11434", Box::new(client))
            .model(Some(MODEL_ID.to_string()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}))
    }

    #[test]
    fn discovers_model_id_from_tags_endpoint() {
        let client = shared_mock(vec![tags_response()]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client.clone()))
            .build()
            .with_backoff(Box::new(|_| {}));
        assert_eq!(emb.model_id().unwrap(), MODEL_ID);
        let reqs = client.borrow().requests();
        assert_eq!(reqs[0].method, "GET");
        assert_eq!(reqs[0].url, "http://localhost:11434/api/tags");
    }

    #[test]
    fn embed_payload_contains_model_and_input() {
        let client = shared_mock(vec![embed_response(2, DIM)]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client.clone()))
            .model(Some(MODEL_ID.to_string()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["a".to_string(), "b".to_string()])
            .unwrap();
        let reqs = client.borrow().requests();
        assert_eq!(reqs[0].url, "http://localhost:11434/api/embed");
        let body: serde_json::Value =
            serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], serde_json::json!(MODEL_ID));
        assert_eq!(body["input"], serde_json::json!(["a", "b"]));
        // Truncation is sent explicitly (default true, matching Python).
        assert_eq!(body["truncate"], serde_json::json!(true));
    }

    #[test]
    fn embed_resolves_model_lazily_when_unconfigured() {
        let client = shared_mock(vec![tags_response(), embed_response(1, DIM)]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client.clone()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["x".to_string()]).unwrap();
        let reqs = client.borrow().requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].url, "http://localhost:11434/api/tags");
        assert_eq!(reqs[1].url, "http://localhost:11434/api/embed");
        let body: serde_json::Value =
            serde_json::from_str(reqs[1].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], serde_json::json!(MODEL_ID));
    }

    #[test]
    fn output_dimension_is_sent_when_configured() {
        let client = shared_mock(vec![embed_response(1, DIM)]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client.clone()))
            .model(Some(MODEL_ID.to_string()))
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
    fn parses_embeddings_field_in_input_order() {
        let client = shared_mock(vec![embed_response(2, DIM)]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client))
            .model(Some(MODEL_ID.to_string()))
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
    fn status_500_is_retried() {
        let client = shared_mock(vec![
            MockResponse::json(500, "server error"),
            embed_response(1, DIM),
        ]);
        let emb = make_embedder(client.clone());
        let vecs = emb.embed_documents(&["x".to_string()]).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(client.borrow().request_count(), 2);
    }

    #[test]
    fn status_404_is_not_retried() {
        let client = shared_mock(vec![MockResponse::json(404, "model not found")]);
        let emb = make_embedder(client.clone());
        let err = emb.embed_documents(&["x".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("HTTP 404"),
            "unexpected error: {err}"
        );
        assert_eq!(client.borrow().request_count(), 1);
    }

    #[test]
    fn fetch_model_id_errors_on_missing_name() {
        let client = shared_mock(vec![MockResponse::json(
            200,
            r#"{"models":[{"model":"x"}]}"#,
        )]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client))
            .build()
            .with_backoff(Box::new(|_| {}));
        let err = emb.model_id().unwrap_err();
        assert!(
            err.to_string().contains("model name"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn embed_errors_on_non_numeric_value() {
        let client = shared_mock(vec![MockResponse::json(
            200,
            r#"{"embeddings":[["oops"]]}"#,
        )]);
        let emb = OllamaBuilder::new("http://localhost:11434", Box::new(client))
            .model(Some(MODEL_ID.to_string()))
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
