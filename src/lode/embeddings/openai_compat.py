"""Embedding client speaking the OpenAI-compatible embeddings API.

Native endpoints used:
    GET  /v1/models       -> {"data": [{"id": "model_id", ...}, ...]}
    POST /v1/embeddings   -> {"model": "...", "input": [...]}
                          -> {"data": [{"embedding": [...], "index": 0}, ...]}

Any OpenAI-compatible server works: TEI, Ollama, vLLM, llama.cpp, hosted
APIs. Those servers expose no `normalize` flag, so L2 normalization happens
client-side via `l2_normalize` in base.py — that is also what makes the
`Embedder` contract ("cosine == dot") hold regardless of backend.

Note: /v1/models reports the model id but not the vector dimension, so
unless a dimension is given explicitly we auto-detect it with a tiny probe
request.

See https://platform.openai.com/docs/api-reference/embeddings
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import httpx

from lode.embeddings.base import Embedder, l2_normalize
from lode.embeddings.errors import EmbedderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080"
# TEI advertises max_client_batch_size=32; other servers accept more, but
# staying at or below it keeps the client safe across backends.
DEFAULT_BATCH_SIZE = 32
# Used to auto-detect the vector dimension at startup.
PROBE_TEXT = "ping"

# Retryable statuses. /v1/embeddings is idempotent, so retries are safe.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleEmbedder(Embedder):
    """Client for any OpenAI-compatible embeddings endpoint.

    Construction is side-effect free: metadata (model id, dimension) is
    resolved lazily on first access, so instantiating an embedder never
    touches the network.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
        retries: int = 3,
        normalize: bool = True,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.normalize = normalize
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)
        self._model = model
        self._dimension = dimension
        self._fetched_model: str | None = None  # resolved lazily from /v1/models

    # -- metadata (lazy) ----------------------------------------------------

    @property
    def model_id(self) -> str:
        if self._model is not None:
            return self._model
        if self._fetched_model is None:
            self._fetched_model = self._fetch_model_id()
        return self._fetched_model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self._embed([PROBE_TEXT])[0])
            logger.info(
                "Struck a lode: model=%s, dimension=%d",
                self.model_id,
                self._dimension,
            )
        return self._dimension

    # -- public API ---------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    # -- internal -----------------------------------------------------------

    def _fetch_model_id(self) -> str:
        resp = self._request("GET", f"{self.base_url}/v1/models")
        data = cast(dict[str, Any], resp.json())
        models = data.get("data", [])
        if not models:
            raise EmbedderUnavailableError("GET /v1/models returned no models")
        model_id = models[0].get("id")
        if not model_id:
            raise EmbedderUnavailableError("GET /v1/models returned a model without an id")
        model_id = str(model_id)
        logger.info("Struck a lode: model=%s", model_id)
        return model_id

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model_id, "input": texts}
        resp = self._request("POST", f"{self.base_url}/v1/embeddings", json=payload)
        data = cast(list[dict[str, Any]], resp.json().get("data", []))
        if len(data) != len(texts):
            raise EmbedderUnavailableError(f"/v1/embeddings returned {len(data)} vectors for {len(texts)} inputs")
        # The spec says vectors come back in input order, but servers may
        # reorder; index is authoritative, so sort by it.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [list(map(float, item["embedding"])) for item in ordered]
        return l2_normalize(vectors) if self.normalize else vectors

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._api_key is not None:
            headers = dict(kwargs.pop("headers", {}))
            headers["Authorization"] = f"Bearer {self._api_key}"
            kwargs["headers"] = headers
        attempt = 0
        while True:
            try:
                resp = self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                if attempt >= self.retries:
                    raise EmbedderUnavailableError(
                        f"embedding endpoint {url} is unreachable after {attempt + 1} attempts: {exc}"
                    ) from exc
                attempt += 1
                logger.warning(
                    "OpenAI-compatible embedding request stumbled (%s); retrying attempt %d/%d",
                    exc,
                    attempt,
                    self.retries,
                )
                self._backoff(attempt)
                continue

            if resp.status_code < 400 or resp.status_code not in RETRYABLE_STATUS or attempt >= self.retries:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise EmbedderUnavailableError(
                        f"embedding endpoint {url} returned HTTP {resp.status_code}: {exc}"
                    ) from exc
                return resp

            attempt += 1
            logger.warning(
                "OpenAI-compatible embedding endpoint stumbled with status %d; retrying attempt %d/%d",
                resp.status_code,
                attempt,
                self.retries,
            )
            self._backoff(attempt)

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2**attempt, 8))  # 2s, 4s, 8s
