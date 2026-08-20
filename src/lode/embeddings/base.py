"""Embedding model interface.

Core modules depend only on this interface, never on a concrete
implementation, so the model backend (local TEI, hosted API, in-process
sentence-transformers) can be swapped via configuration.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    """Interface for embedding models.

    All implementations:
      - expose `model_id` and `dimension` (resolved lazily; construction is
        side-effect free, so creating an embedder never touches the network)
      - embed a list of documents (bulk) or a single query string
      - return L2-normalized vectors when `normalize` is enabled, so that
        cosine similarity == dot product and consumers can rank by dot
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier. May trigger a metadata request on first access."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension. May trigger a probe request on first access."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Empty input returns an empty list."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


def l2_normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """L2-normalize each vector, returning new lists (inputs are untouched).

    Zero vectors are returned as-is (their norm is 0; dividing by it would
    produce NaN). Re-normalizing an already-normalized vector is a no-op up
    to floating-point error, so implementations may call this unconditionally
    as a safety net.
    """
    result: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            result.append(list(vector))
        else:
            scale = 1.0 / norm
            result.append([x * scale for x in vector])
    return result
