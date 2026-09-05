#![warn(clippy::pedantic)]

//! Test-only HTTP mock, mirroring Python's `httpx.MockTransport`.
//!
//! Responses are scripted in order; each `request` call pops the next one,
//! which makes retry scenarios natural (script a transient status, then a
//! success). All outgoing requests are recorded for assertions.

use std::cell::RefCell;
use std::collections::VecDeque;

use super::http::{HttpClient, HttpError, HttpResponse};

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
