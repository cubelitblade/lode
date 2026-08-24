"""Tests for the OpenAI-compatible embedding client.

A fake OpenAI-compatible server is built on httpx.MockTransport, so these
tests are hermetic and fast. The fake records /v1/embeddings payloads so
tests can assert on batching, the model field, and request order.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from lode.embeddings.errors import EmbedderUnavailableError
from lode.embeddings.openai_compat import OpenAICompatibleEmbedder

DIM = 512
MODEL_ID = "BAAI/bge-small-zh-v1.5"


class _FakeServer:
    """In-memory OpenAI-compatible server recording /v1/embeddings payloads."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.embed_calls: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": MODEL_ID, "object": "model", "owned_by": "fake"}],
                },
            )
        if request.url.path == "/v1/embeddings":
            if self.fail_times > 0:
                self.fail_times -= 1
                return httpx.Response(503, json={"error": "overloaded"})
            payload = json.loads(request.content)
            self.embed_calls.append(payload)
            inputs = payload["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [float(i) + 1 for _ in range(DIM)],
                        }
                        for i in range(len(inputs))
                    ],
                    "model": payload.get("model", MODEL_ID),
                },
            )
        return httpx.Response(404, text="not found")


def _make_embedder(fake: _FakeServer, **kwargs: Any) -> OpenAICompatibleEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), timeout=5.0)
    return OpenAICompatibleEmbedder(client=client, **kwargs)


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def test_auto_detect_model_id_and_dimension() -> None:
    emb = _make_embedder(_FakeServer())
    assert emb.model_id == MODEL_ID
    assert emb.dimension == DIM


def test_explicit_model_and_dimension_skip_requests() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM)
    assert emb.model_id == MODEL_ID
    assert emb.dimension == DIM
    assert fake.embed_calls == []  # no probe or metadata request was sent


def test_construction_is_side_effect_free() -> None:
    # The fake raises if any request reaches it; construction must not probe.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    OpenAICompatibleEmbedder(client=client)  # must not raise


def test_embed_documents() -> None:
    emb = _make_embedder(_FakeServer(), dimension=DIM)
    vecs = emb.embed_documents(["a", "b"])
    assert len(vecs) == 2
    assert all(len(v) == DIM for v in vecs)


def test_embed_query_single() -> None:
    emb = _make_embedder(_FakeServer(), dimension=DIM)
    v = emb.embed_query("hello")
    assert len(v) == DIM


def test_empty_input_makes_no_request() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, dimension=DIM)
    assert emb.embed_documents([]) == []
    assert fake.embed_calls == []


def test_batching_respects_batch_size() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, dimension=DIM, batch_size=2)
    vecs = emb.embed_documents(["a", "b", "c", "d", "e"])
    assert len(vecs) == 5
    assert [len(c["input"]) for c in fake.embed_calls] == [2, 2, 1]


def test_model_field_sent_in_payload() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM)
    emb.embed_documents(["x"])
    assert fake.embed_calls[0]["model"] == MODEL_ID


def test_normalized_by_default() -> None:
    emb = _make_embedder(_FakeServer(), dimension=DIM)
    vecs = emb.embed_documents(["x", "y"])
    assert all(math.isclose(_norm(v), 1.0, rel_tol=1e-6) for v in vecs)


def test_normalize_can_be_disabled() -> None:
    emb = _make_embedder(_FakeServer(), dimension=DIM, normalize=False)
    v = emb.embed_query("x")
    # raw vector is [1.0, 1.0, ...], norm == sqrt(DIM), far from 1.0
    assert math.isclose(_norm(v), math.sqrt(DIM), rel_tol=1e-6)


def test_l2_normalize_is_idempotent() -> None:
    emb = _make_embedder(_FakeServer(), dimension=DIM)
    vecs = emb.embed_documents(["x"])
    again = emb.embed_documents(["x"])
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(vecs[0], again[0], strict=True))


def test_vectors_reordered_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [1.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 4.0]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OpenAICompatibleEmbedder(client=client, dimension=2, normalize=False)
    v0, v1 = emb.embed_documents(["a", "b"])
    assert v0 == [3.0, 4.0]
    assert v1 == [1.0, 2.0]


def test_retries_then_succeeds() -> None:
    fake = _FakeServer(fail_times=1)
    emb = _make_embedder(fake, dimension=DIM, retries=3)
    vecs = emb.embed_documents(["x"])
    assert len(vecs) == 1
    assert len(fake.embed_calls) == 1


def test_gives_up_after_max_retries() -> None:
    fake = _FakeServer(fail_times=99)
    emb = _make_embedder(fake, dimension=DIM, retries=2)
    with pytest.raises(EmbedderUnavailableError):
        emb.embed_documents(["x"])
    assert fake.embed_calls == []


def test_network_failure_wrapped_as_embedder_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OpenAICompatibleEmbedder(client=client, dimension=DIM, retries=1)
    with pytest.raises(EmbedderUnavailableError):
        emb.embed_query("x")


def test_wrong_vector_count_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
        data = [{"index": 0, "embedding": [0.1, 0.2]}]
        return httpx.Response(200, json={"data": data})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OpenAICompatibleEmbedder(client=client, dimension=2)
    with pytest.raises(ValueError, match="vectors"):
        emb.embed_documents(["a", "b"])


def test_empty_model_list_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OpenAICompatibleEmbedder(client=client)
    with pytest.raises(ValueError, match="no models"):
        _ = emb.model_id


def test_api_key_sends_bearer_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
        if request.url.path == "/v1/embeddings":
            seen["authorization"] = request.headers.get("Authorization", "")
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0] * DIM}], "model": MODEL_ID},
            )
        return httpx.Response(404, text="not found")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OpenAICompatibleEmbedder(client=client, model=MODEL_ID, dimension=DIM, api_key="secret")
    emb.embed_documents(["x"])
    assert seen["authorization"] == "Bearer secret"
