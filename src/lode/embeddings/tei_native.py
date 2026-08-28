"""Embedding client speaking the TEI native API.

Native endpoints used:
    GET  /info        -> {"model_id": "...", "max_client_batch_size": 32, ...}
    POST /embed       -> {"inputs": [...], "normalize": true, ...}
                     -> [[...], [...]]  (bare vectors, in input order)

Unlike the OpenAI-compatible endpoint, TEI's native `/embed` returns a bare
array of vectors in input order (no `index`/`model` wrapper), and `/info`
reports the model id but not the vector dimension, so unless a dimension is
given explicitly we auto-detect it with a tiny probe request.

`normalize` is applied client-side via `l2_normalize` in base.py to keep the
`Embedder` contract ("cosine == dot") centralized, matching the
OpenAI-compatible backend.

See https://huggingface.github.io/text-embeddings-inference/openapi
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
DEFAULT_BATCH_SIZE = 32
# Used to auto-detect the vector dimension at startup.
PROBE_TEXT = "ping"

# Retryable statuses. 424 (backend inference failure) is NOT retryable: the
# request reached the model and failed deterministically, so retrying would
# not help. 429 (overloaded) is safe to retry.
RETRYABLE_STATUS = {429}


class HuggingFaceTEINativeEmbedder(Embedder):
    """Client for the TEI native embeddings API.

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
        output_dimension: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
        retries: int = 3,
        normalize: bool = True,
        truncate: bool = False,
        truncation_direction: str = "right",
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.normalize = normalize
        self.truncate = truncate
        self.truncation_direction = truncation_direction
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)
        self._model = model
        self._dimension = dimension
        self._output_dimension = output_dimension
        self._fetched_model: str | None = None  # resolved lazily from /info

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
        resp = self._request("GET", f"{self.base_url}/info")
        data = cast(dict[str, Any], resp.json())
        model_id = data.get("model_id")
        if not model_id:
            raise EmbedderUnavailableError("GET /info returned no model_id")
        model_id = str(model_id)
        logger.info("Struck a lode: model=%s", model_id)
        return model_id

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "inputs": texts,
            "normalize": self.normalize,
            "truncate": self.truncate,
            "truncation_direction": self.truncation_direction,
        }
        if self._output_dimension is not None:
            payload["dimensions"] = self._output_dimension
        resp = self._request("POST", f"{self.base_url}/embed", json=payload)
        data = cast(list[list[float]], resp.json())
        if len(data) != len(texts):
            raise EmbedderUnavailableError(f"/embed returned {len(data)} vectors for {len(texts)} inputs")
        # TEI returns bare vectors in input order; no index to sort by.
        vectors = [list(map(float, item)) for item in data]
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
                    "TEI native embedding request stumbled (%s); retrying attempt %d/%d",
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
                "TEI native embedding endpoint stumbled with status %d; retrying attempt %d/%d",
                resp.status_code,
                attempt,
                self.retries,
            )
            self._backoff(attempt)

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2**attempt, 8))  # 2s, 4s, 8s
