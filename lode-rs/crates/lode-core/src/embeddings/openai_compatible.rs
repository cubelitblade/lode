#![warn(clippy::pedantic)]

//! OpenAI-compatible embedding API.
//!
//! Endpoints:
//! - `GET {base}/models`       -> discovers the served model id
//! - `POST {base}/embeddings`  -> embeds batches
//!
//! Responses nest vectors under `data[*].embedding` and may arrive unordered,
//! so we sort by `index` before returning.

use super::{
    base::Embedder,
    errors,
    http::{HttpBackend, HttpEmbedder},
};

const PATH_MODELS: &str = "/v1/models";
const PATH_EMBEDDINGS: &str = "/v1/embeddings";

/// Transient statuses worth retrying.
const RETRYABLE_STATUS: &[u16] = &[429, 500, 502, 503, 504];

/// Builder producing an OpenAI-compatible `HttpEmbedder`.
pub struct OpenAiCompatibleBuilder<'a> {
    base_url: &'a str,
    model: Option<String>,
    dimension: Option<usize>,
    output_dimension: Option<usize>,
    batch_size: usize,
    retries: u32,
    api_key: Option<String>,
    client: Box<dyn super::http::HttpClient>,
}

impl<'a> OpenAiCompatibleBuilder<'a> {
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
    pub fn build(self) -> HttpEmbedder {
        HttpEmbedder::new(
            Box::new(OpenAiCompatibleBackend),
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

struct OpenAiCompatibleBackend;

impl HttpBackend for OpenAiCompatibleBackend {
    fn backend_label(&self) -> &'static str {
        "OpenAI-compatible"
    }

    fn retryable_status(&self) -> &'static [u16] {
        RETRYABLE_STATUS
    }

    fn fetch_model_id(&self, http: &HttpEmbedder) -> crate::Result<String> {
        let url = format!("{}{PATH_MODELS}", http.base_url());
        let resp = http.request("GET", &url, None)?;
        let parsed: serde_json::Value = serde_json::from_str(&resp.body)
            .map_err(|e| errors::err(format!("invalid JSON from {url}: {e}")))?;
        let arr = parsed
            .get("data")
            .and_then(|v| v.as_array())
            .filter(|arr| !arr.is_empty())
            .ok_or_else(|| errors::missing_field("non-empty data[]", &url))?;
        let id = arr[0]
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| errors::missing_field("model id", &url))?;
        Ok(id.to_string())
    }

    fn embed(&self, http: &HttpEmbedder, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        let url = format!("{}{PATH_EMBEDDINGS}", http.base_url());

        let inputs: Vec<serde_json::Value> = texts.iter().map(|t| t.clone().into()).collect();
        let mut map = serde_json::Map::new();
        map.insert("input".into(), serde_json::Value::Array(inputs));
        // The OpenAI-compatible API requires `model`; resolve it lazily when
        // not configured (matches Python, which always sends it).
        map.insert("model".into(), http.model_id()?.into());
        if let Some(out_dim) = http.output_dimension() {
            map.insert("dimensions".into(), out_dim.into());
        }

        let payload = serde_json::Value::Object(map).to_string();
        let resp = http.request("POST", &url, Some(&payload))?;
        let parsed: serde_json::Value = serde_json::from_str(&resp.body)
            .map_err(|e| errors::err(format!("invalid JSON from {url}: {e}")))?;

        let data = parsed
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| errors::missing_field("data[]", &url))?;

        let mut ordered: Vec<(i64, Vec<f32>)> = Vec::with_capacity(data.len());
        for item in data {
            let idx = item
                .get("index")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(i64::MIN);
            let vec_raw = item
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| errors::missing_field("embedding", &url))?;
            let floats = vec_raw
                .iter()
                .map(|v| v.as_f64().map(crate::embeddings::embedding_scalar_to_f32))
                .collect::<Option<Vec<f32>>>()
                .ok_or_else(|| errors::err(format!("non-numeric embedding value from {url}")))?;
            ordered.push((idx, floats));
        }
        ordered.sort_by_key(|(idx, _)| *idx);
        Ok(ordered.into_iter().map(|(_, v)| v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::base::Embedder;
    use crate::embeddings::testing::{MockHttpClientHandle, MockResponse, shared_mock};

    const MODEL_ID: &str = "test-embedding-model";
    const DIM: usize = 3;

    fn models_response() -> MockResponse {
        MockResponse::json(200, &format!(r#"{{"data":[{{"id":"{MODEL_ID}"}}]}}"#))
    }

    fn embed_response(texts: usize, dim: usize) -> MockResponse {
        let data: Vec<serde_json::Value> = (0..texts)
            .map(|i| {
                serde_json::json!({
                    "index": i,
                    "embedding": vec![1.0; dim],
                })
            })
            .collect();
        MockResponse::json(200, &serde_json::json!({ "data": data }).to_string())
    }

    fn make_embedder(client: MockHttpClientHandle) -> HttpEmbedder {
        OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client))
            .model(Some(MODEL_ID.to_string()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}))
    }

    #[test]
    fn discovers_model_id_from_models_endpoint() {
        let client = shared_mock(vec![models_response()]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .build()
            .with_backoff(Box::new(|_| {}));
        assert_eq!(emb.model_id().unwrap(), MODEL_ID);
        let reqs = client.borrow().requests();
        assert_eq!(reqs[0].method, "GET");
        assert_eq!(reqs[0].url, "http://localhost:8080/v1/models");
    }

    #[test]
    fn empty_model_list_raises() {
        let client = shared_mock(vec![MockResponse::json(200, r#"{"data":[]}"#)]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client))
            .build()
            .with_backoff(Box::new(|_| {}));
        let err = emb.model_id().unwrap_err();
        assert!(err.to_string().contains("data"), "unexpected error: {err}");
    }

    #[test]
    fn embed_payload_contains_input_and_model() {
        let client = shared_mock(vec![embed_response(2, DIM)]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .model(Some(MODEL_ID.to_string()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["a".to_string(), "b".to_string()])
            .unwrap();
        let reqs = client.borrow().requests();
        assert_eq!(reqs[0].url, "http://localhost:8080/v1/embeddings");
        let body: serde_json::Value =
            serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["input"], serde_json::json!(["a", "b"]));
        assert_eq!(body["model"], serde_json::json!(MODEL_ID));
    }

    #[test]
    fn embed_resolves_model_lazily_when_unconfigured() {
        let client = shared_mock(vec![models_response(), embed_response(1, DIM)]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .dimension(Some(DIM))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["x".to_string()]).unwrap();
        let reqs = client.borrow().requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].url, "http://localhost:8080/v1/models");
        assert_eq!(reqs[1].url, "http://localhost:8080/v1/embeddings");
        let body: serde_json::Value =
            serde_json::from_str(reqs[1].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], serde_json::json!(MODEL_ID));
    }

    #[test]
    fn output_dimension_is_sent_when_configured() {
        let client = shared_mock(vec![embed_response(1, DIM)]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client.clone()))
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
    fn vectors_reordered_by_index() {
        let client = shared_mock(vec![MockResponse::json(
            200,
            r#"{"data":[{"index":1,"embedding":[0.0,1.0]},{"index":0,"embedding":[1.0,0.0]}]}"#,
        )]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client))
            .model(Some(MODEL_ID.to_string()))
            .dimension(Some(2))
            .build()
            .with_backoff(Box::new(|_| {}));
        let vecs = emb
            .embed_documents(&["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(vecs[0], vec![1.0, 0.0]);
        assert_eq!(vecs[1], vec![0.0, 1.0]);
    }

    #[test]
    fn retries_on_429_then_succeeds() {
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
    fn api_key_sends_bearer_header() {
        let client = shared_mock(vec![embed_response(1, DIM)]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client.clone()))
            .model(Some(MODEL_ID.to_string()))
            .dimension(Some(DIM))
            .api_key(Some("secret".to_string()))
            .build()
            .with_backoff(Box::new(|_| {}));
        emb.embed_documents(&["x".to_string()]).unwrap();
        let reqs = client.borrow().requests();
        assert!(
            reqs[0]
                .headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer secret")
        );
    }

    #[test]
    fn fetch_model_id_errors_on_missing_id() {
        let client = shared_mock(vec![MockResponse::json(200, r#"{"data":[{"name":"x"}]}"#)]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client))
            .build()
            .with_backoff(Box::new(|_| {}));
        let err = emb.model_id().unwrap_err();
        assert!(
            err.to_string().contains("model id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn embed_errors_on_non_numeric_value() {
        let client = shared_mock(vec![MockResponse::json(
            200,
            r#"{"data":[{"index":0,"embedding":["oops"]}]}"#,
        )]);
        let emb = OpenAiCompatibleBuilder::new("http://localhost:8080", Box::new(client))
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
