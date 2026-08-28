"""Tests for the Ollama native embedding client.

A fake Ollama server is built on httpx.MockTransport, so these tests are
hermetic and fast. The fake records /api/embed payloads so tests can assert
on batching, the truncate/dimensions fields, and request order.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from lode.embeddings.errors import EmbedderUnavailableError
from lode.embeddings.ollama import OllamaEmbedder

DIM = 512
MODEL_ID = "all-minilm"


class _FakeServer:
    """In-memory Ollama server recording /api/embed payloads."""

    def __init__(
        self,
        *,
        fail_times: int = 0,
        fail_status: int = 500,
        models: list[dict[str, Any]] | None = None,
    ) -> None:
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.models = models if models is not None else [{"name": MODEL_ID}]
        self.embed_calls: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": self.models})
        if request.url.path == "/api/embed":
            if self.fail_times > 0:
                self.fail_times -= 1
                return httpx.Response(self.fail_status, json={"error": "overloaded"})
            payload = json.loads(request.content)
            self.embed_calls.append(payload)
            inputs = payload["input"]
            # Ollama returns bare vectors in input order (no index wrapper).
            return httpx.Response(
                200,
                json={
                    "model": payload["model"],
                    "embeddings": [[float(i) + 1 for _ in range(DIM)] for i in range(len(inputs))],
                },
            )
        return httpx.Response(404, text="not found")


def _make_embedder(fake: _FakeServer, **kwargs: Any) -> OllamaEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), timeout=5.0)
    return OllamaEmbedder(client=client, **kwargs)


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def test_auto_detect_model_id_and_dimension() -> None:
    emb = _make_embedder(_FakeServer())
    assert emb.model_id == MODEL_ID
    assert emb.dimension == DIM


def test_auto_detect_picks_first_model_in_list() -> None:
    # Heuristic discovery: the first entry wins, chat models included.
    fake = _FakeServer(models=[{"name": "llama3.2"}, {"name": MODEL_ID}])
    emb = _make_embedder(fake)
    assert emb.model_id == "llama3.2"


def test_auto_detect_with_empty_model_list_raises() -> None:
    emb = _make_embedder(_FakeServer(models=[]))
    with pytest.raises(EmbedderUnavailableError, match="no models"):
        _ = emb.model_id


def test_explicit_model_and_dimension_skip_requests() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM)
    assert emb.model_id == MODEL_ID
    assert emb.dimension == DIM
    assert fake.embed_calls == []  # no probe request was sent


def test_construction_is_side_effect_free() -> None:
    # The fake raises if any request reaches it; construction must not probe.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    OllamaEmbedder(client=client, model=MODEL_ID)  # must not raise


def test_embed_documents() -> None:
    emb = _make_embedder(_FakeServer(), model=MODEL_ID, dimension=DIM)
    vecs = emb.embed_documents(["a", "b"])
    assert len(vecs) == 2
    assert all(len(v) == DIM for v in vecs)


def test_embed_query_single() -> None:
    emb = _make_embedder(_FakeServer(), model=MODEL_ID, dimension=DIM)
    v = emb.embed_query("hello")
    assert len(v) == DIM


def test_empty_input_makes_no_request() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM)
    assert emb.embed_documents([]) == []
    assert fake.embed_calls == []


def test_batching_respects_batch_size() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM, batch_size=2)
    vecs = emb.embed_documents(["a", "b", "c", "d", "e"])
    assert len(vecs) == 5
    assert [len(c["input"]) for c in fake.embed_calls] == [2, 2, 1]


def test_payload_fields_sent() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM)
    emb.embed_documents(["x"])
    call = fake.embed_calls[0]
    assert call["model"] == MODEL_ID
    assert call["input"] == ["x"]
    # Ollama's server-side default is truncate=true; the client sends it
    # explicitly so behaviour does not depend on server defaults.
    assert call["truncate"] is True


def test_truncate_can_be_disabled() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM, truncate=False)
    emb.embed_documents(["x"])
    assert fake.embed_calls[0]["truncate"] is False


def test_normalized_by_default() -> None:
    emb = _make_embedder(_FakeServer(), model=MODEL_ID, dimension=DIM)
    vecs = emb.embed_documents(["x", "y"])
    assert all(math.isclose(_norm(v), 1.0, rel_tol=1e-6) for v in vecs)


def test_normalize_can_be_disabled() -> None:
    emb = _make_embedder(_FakeServer(), model=MODEL_ID, dimension=DIM, normalize=False)
    v = emb.embed_query("x")
    # raw vector is [1.0, 1.0, ...], norm == sqrt(DIM), far from 1.0
    assert math.isclose(_norm(v), math.sqrt(DIM), rel_tol=1e-6)


def test_vectors_kept_in_input_order() -> None:
    # Ollama returns bare vectors in input order; the client must not reorder.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0], [3.0, 4.0]]})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OllamaEmbedder(client=client, model=MODEL_ID, dimension=2, normalize=False)
    v0, v1 = emb.embed_documents(["a", "b"])
    assert v0 == [1.0, 2.0]
    assert v1 == [3.0, 4.0]


def test_output_dimension_sent_in_payload() -> None:
    fake = _FakeServer()
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM, output_dimension=256)
    emb.embed_documents(["x"])
    assert fake.embed_calls[0]["dimensions"] == 256


def test_retries_then_succeeds() -> None:
    fake = _FakeServer(fail_times=1)
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM, retries=3)
    vecs = emb.embed_documents(["x"])
    assert len(vecs) == 1
    assert len(fake.embed_calls) == 1


def test_gives_up_after_max_retries() -> None:
    fake = _FakeServer(fail_times=99)
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM, retries=2)
    with pytest.raises(EmbedderUnavailableError):
        emb.embed_documents(["x"])
    assert fake.embed_calls == []


def test_404_is_not_retried() -> None:
    # 404 (model not found) is deterministic and must not be retried.
    fake = _FakeServer(fail_times=99, fail_status=404)
    emb = _make_embedder(fake, model=MODEL_ID, dimension=DIM, retries=3)
    with pytest.raises(EmbedderUnavailableError):
        emb.embed_documents(["x"])
    assert fake.embed_calls == []


def test_vector_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OllamaEmbedder(client=client, model=MODEL_ID, dimension=2)
    with pytest.raises(EmbedderUnavailableError, match="vectors for 2 inputs"):
        emb.embed_documents(["a", "b"])


def test_network_failure_wrapped_as_embedder_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    emb = OllamaEmbedder(client=client, model=MODEL_ID, dimension=DIM, retries=1)
    with pytest.raises(EmbedderUnavailableError, match="unreachable"):
        emb.embed_documents(["x"])
