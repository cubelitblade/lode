"""Shared HTTP plumbing for embedding clients that speak a REST API.

Owns everything both concrete backends need: lazy metadata resolution
(`model_id` / `dimension`), batched embedding, retry with exponential
backoff, and API-key headers. Subclasses only describe what differs:

  - `RETRYABLE_STATUS`: which HTTP statuses are safe to retry
  - `_backend_label`: backend tag used in log messages
  - `_fetch_model_id()`: how to discover the model id
  - `_embed(texts)`: payload construction and response parsing

Construction stays side-effect free: metadata is resolved lazily on first
access, so instantiating an embedder never touches the network.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx

from lode.embeddings.base import Embedder
from lode.embeddings.errors import EmbedderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080"
# TEI advertises max_client_batch_size=32; other servers accept more, but
# staying at or below it keeps the client safe across backends.
DEFAULT_BATCH_SIZE = 32
# Used to auto-detect the vector dimension at startup.
PROBE_TEXT = "ping"


class HttpEmbedder(Embedder, ABC):
    """Base class for embedding clients over a REST API.

    Construction is side-effect free: metadata (model id, dimension) is
    resolved lazily on first access, so instantiating an embedder never
    touches the network.
    """

    # HTTP statuses safe to retry; subclasses narrow this per backend.
    RETRYABLE_STATUS: ClassVar[frozenset[int]]
    # Backend tag used in log messages (e.g. "OpenAI-compatible").
    _backend_label: ClassVar[str]

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str | None = None,
        dimension: int | None = None,
        output_dimension: int | None = None,
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
        self._output_dimension = output_dimension
        self._fetched_model: str | None = None

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

    @abstractmethod
    def _fetch_model_id(self) -> str: ...

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]: ...

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
                    "%s embedding request stumbled (%s); retrying attempt %d/%d",
                    self._backend_label,
                    exc,
                    attempt,
                    self.retries,
                )
                self._backoff(attempt)
                continue

            if resp.status_code < 400 or resp.status_code not in self.RETRYABLE_STATUS or attempt >= self.retries:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise EmbedderUnavailableError(
                        f"embedding endpoint {url} returned HTTP {resp.status_code}: {exc}"
                    ) from exc
                return resp

            attempt += 1
            logger.warning(
                "%s embedding endpoint stumbled with status %d; retrying attempt %d/%d",
                self._backend_label,
                resp.status_code,
                attempt,
                self.retries,
            )
            self._backoff(attempt)

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2**attempt, 8))  # 2s, 4s, 8s
