#![warn(clippy::pedantic)]

//! Test-only HTTP mock, mirroring Python's `httpx.MockTransport`.
//!
//! Responses are scripted in order; each `request` call pops the next one,
//! which makes retry scenarios natural (script a transient status, then a
//! success). All outgoing requests are recorded for assertions.

use std::cell::RefCell;
use std::collections::VecDeque;

use super::base::Embedder;
use super::http::{HttpClient, HttpError, HttpResponse};
use crate::index::records::{FileRecord, FileStatus};
use crate::ingestion::digest::chunk_digest;
use crate::ingestion::types::Chunk;
use crate::relpath::WorkspacePath;

/// A recorded outgoing request.
#[derive(Debug, Clone)]
pub struct MockRequest {
    pub method: String,
    pub url: String,
    pub body: Option<String>,
    pub headers: Vec<(String, String)>,
}

/// A scripted response.
#[derive(Debug, Clone)]
pub struct MockResponse {
    pub status: u16,
    pub body: String,
}

impl MockResponse {
    #[must_use]
    pub fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_string(),
        }
    }
}

/// Scripted `HttpClient` for hermetic tests.
pub struct MockHttpClient {
    responses: RefCell<VecDeque<MockResponse>>,
    requests: RefCell<Vec<MockRequest>>,
}

impl MockHttpClient {
    #[must_use]
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: RefCell::new(responses.into()),
            requests: RefCell::new(Vec::new()),
        }
    }

    /// All requests issued so far, in order.
    pub fn requests(&self) -> Vec<MockRequest> {
        self.requests.borrow().clone()
    }

    /// Number of requests issued so far.
    pub fn request_count(&self) -> usize {
        self.requests.borrow().len()
    }
}

impl HttpClient for MockHttpClient {
    fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
        headers: &[(String, String)],
    ) -> Result<HttpResponse, HttpError> {
        self.requests.borrow_mut().push(MockRequest {
            method: method.to_string(),
            url: url.to_string(),
            body: body.map(str::to_string),
            headers: headers.to_vec(),
        });
        let next = self
            .responses
            .borrow_mut()
            .pop_front()
            .expect("MockHttpClient ran out of scripted responses");
        Ok(HttpResponse {
            status: next.status,
            body: next.body,
        })
    }
}

/// Shared handle so tests can inspect requests after handing the client to an
/// embedder (which owns it behind `Box<dyn HttpClient>`).
pub type MockHttpClientHandle = std::rc::Rc<std::cell::RefCell<MockHttpClient>>;

impl HttpClient for MockHttpClientHandle {
    fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
        headers: &[(String, String)],
    ) -> Result<HttpResponse, HttpError> {
        self.borrow().request(method, url, body, headers)
    }
}

/// Convenience: build a shared mock with the given scripted responses.
#[must_use]
pub fn shared_mock(responses: Vec<MockResponse>) -> MockHttpClientHandle {
    std::rc::Rc::new(std::cell::RefCell::new(MockHttpClient::new(responses)))
}

/// A deterministic embedder for tests: no network, no drift.
///
/// Mirrors Python's `FakeEmbedder`: document i embeds to `0.1*(i+1)` repeated
/// across the dimension; every query embeds to a constant vector, so dense
/// scores only depend on the stored vectors (documents with smaller seq are
/// nearer to the query). `query_calls` records `embed_query` invocations.
#[derive(Debug, Clone)]
pub struct FakeEmbedder {
    model: String,
    dim: usize,
    /// Count of `embed_query` calls, for tests that assert embed frequency.
    pub query_calls: std::cell::Cell<usize>,
}

impl FakeEmbedder {
    /// A fake embedder with the given model id and dimension.
    #[must_use]
    pub fn new(model: &str, dimension: usize) -> Self {
        Self {
            model: model.to_string(),
            dim: dimension,
            query_calls: std::cell::Cell::new(0),
        }
    }

    /// The fake dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

impl Default for FakeEmbedder {
    fn default() -> Self {
        Self::new("test-model", 4)
    }
}

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> crate::Result<String> {
        Ok(self.model.clone())
    }

    fn dimension(&self) -> crate::Result<usize> {
        Ok(self.dim)
    }

    fn embed_documents(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                #[expect(clippy::cast_precision_loss, reason = "test batch indices are tiny")]
                let value = 0.1 * (i + 1) as f32;
                vec![value; self.dim]
            })
            .collect())
    }

    fn embed_query(&self, _text: &str) -> crate::Result<Vec<f32>> {
        self.query_calls.set(self.query_calls.get() + 1);
        Ok(vec![0.1; self.dim])
    }
}

/// Test chunk factory mirroring Python's `make_chunks`.
///
/// Chunk i embeds to the vector `[0.1*(i+1), 0.2, 0.3, 0.4]`, so different
/// chunks land at different distances from the fake query vector.
#[must_use]
pub fn make_chunks(texts: &[&str], pages: Option<&[u32]>) -> (Vec<Chunk>, Vec<Vec<f32>>) {
    let chunks = texts
        .iter()
        .enumerate()
        .map(|(seq, &text)| {
            #[expect(clippy::cast_possible_truncation, reason = "tiny chunk count")]
            let seq = seq as u32;
            Chunk {
                digest: chunk_digest(text),
                text: text.to_string(),
                seq,
                heading: String::new(),
                page: pages.and_then(|p| p.get(seq as usize).copied()),
            }
        })
        .collect();
    let vectors = (0..texts.len())
        .map(|seq| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "tiny test indices; exact float value is irrelevant"
            )]
            let leading = 0.1 * (seq + 1) as f32;
            vec![leading, 0.2, 0.3, 0.4]
        })
        .collect();
    (chunks, vectors)
}

/// Test record factory mirroring Python's `file_record`.
#[must_use]
pub fn file_record(path: &str, digest: &str, size: u64) -> FileRecord {
    FileRecord {
        path: WorkspacePath::from_posix(path),
        digest: digest.to_string(),
        mtime: 1.0,
        size,
        status: FileStatus::Fresh,
    }
}
