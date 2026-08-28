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
from typing import Any, cast

import httpx

from lode.embeddings.base import l2_normalize
from lode.embeddings.errors import EmbedderUnavailableError
from lode.embeddings.http import DEFAULT_BASE_URL, DEFAULT_BATCH_SIZE, HttpEmbedder

logger = logging.getLogger(__name__)

# 424 (backend inference failure) is NOT retryable: the request reached the
# model and failed deterministically, so retrying would not help. 429
# (overloaded) is safe to retry.
RETRYABLE_STATUS = frozenset({429})


class HuggingFaceTEINativeEmbedder(HttpEmbedder):
    """Client for the TEI native embeddings API."""

    RETRYABLE_STATUS = RETRYABLE_STATUS
    _backend_label = "TEI native"

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
        self.truncation_direction = truncation_direction

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
