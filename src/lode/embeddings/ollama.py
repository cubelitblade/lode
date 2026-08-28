"""Embedding client speaking the Ollama native API.

Native endpoints used:
    POST /api/embed  -> {"model": "...", "input": [...], "truncate": true}
                     -> {"model": "...", "embeddings": [[...], ...]}

Unlike the OpenAI-compatible endpoint, Ollama's native `/api/embed` natively
supports `truncate` and `dimensions`, and returns bare vectors in input order
(no `index` wrapper) — the same shape as TEI's `/embed`.

Ollama's model listing (`GET /api/tags`) mixes chat and embedding models, so
auto-discovery is heuristic: it probes the first model in the list. On
multi-model servers, set `model` explicitly.

`normalize` is applied client-side via `l2_normalize` in base.py to keep the
`Embedder` contract ("cosine == dot") centralized, matching the other
backends.

See https://docs.ollama.com/api
"""

from __future__ import annotations

import logging
from typing import Any, cast

import httpx

from lode.embeddings.base import l2_normalize
from lode.embeddings.errors import EmbedderUnavailableError
from lode.embeddings.http import DEFAULT_BATCH_SIZE, HttpEmbedder

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

# 404 (model not found) is deterministic and must not be retried; only
# transient server-side failures are.
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


class OllamaEmbedder(HttpEmbedder):
    """Client for the Ollama native embeddings API."""

    RETRYABLE_STATUS = RETRYABLE_STATUS
    _backend_label = "Ollama"

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
        truncate: bool = True,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            base_url,
            model=model,
            dimension=dimension,
            output_dimension=output_dimension,
            batch_size=batch_size,
            timeout=timeout,
            retries=retries,
            normalize=normalize,
            api_key=api_key,
            client=client,
        )
        self.truncate = truncate

    def _fetch_model_id(self) -> str:
        resp = self._request("GET", f"{self.base_url}/api/tags")
        data = cast(dict[str, Any], resp.json())
        models = data.get("models", [])
        if not models:
            raise EmbedderUnavailableError("GET /api/tags returned no models")
        model_id = models[0].get("name")
        if not model_id:
            raise EmbedderUnavailableError("GET /api/tags returned a model without a name")
        model_id = str(model_id)
        logger.info("Struck a lode: model=%s", model_id)
        return model_id

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": texts,
            "truncate": self.truncate,
        }
        if self._output_dimension is not None:
            payload["dimensions"] = self._output_dimension
        resp = self._request("POST", f"{self.base_url}/api/embed", json=payload)
        data = cast(dict[str, Any], resp.json())
        embeddings = cast(list[list[float]], data.get("embeddings") or [])
        if len(embeddings) != len(texts):
            raise EmbedderUnavailableError(f"/api/embed returned {len(embeddings)} vectors for {len(texts)} inputs")
        # Ollama returns bare vectors in input order; no index to sort by.
        vectors = [list(map(float, item)) for item in embeddings]
        return l2_normalize(vectors) if self.normalize else vectors
