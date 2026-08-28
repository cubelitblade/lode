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
from typing import Any, cast

from lode.embeddings.base import l2_normalize
from lode.embeddings.errors import EmbedderUnavailableError
from lode.embeddings.http import HttpEmbedder

logger = logging.getLogger(__name__)

# /v1/embeddings is idempotent, so retries are safe.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class OpenAICompatibleEmbedder(HttpEmbedder):
    """Client for any OpenAI-compatible embeddings endpoint."""

    RETRYABLE_STATUS = RETRYABLE_STATUS
    _backend_label = "OpenAI-compatible"

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
        if self._output_dimension is not None:
            payload["dimensions"] = self._output_dimension
        resp = self._request("POST", f"{self.base_url}/v1/embeddings", json=payload)
        data = cast(list[dict[str, Any]], resp.json().get("data", []))
        if len(data) != len(texts):
            raise EmbedderUnavailableError(f"/v1/embeddings returned {len(data)} vectors for {len(texts)} inputs")
        # The spec says vectors come back in input order, but servers may
        # reorder; index is authoritative, so sort by it.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [list(map(float, item["embedding"])) for item in ordered]
        return l2_normalize(vectors) if self.normalize else vectors
