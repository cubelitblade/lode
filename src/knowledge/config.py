"""Runtime configuration.

Config is the composition root: the rest of the code depends on interfaces
(Embedder, Splitter, ...) and this module decides which concrete
implementation is selected.
"""

from __future__ import annotations

from pydantic import BaseModel

from knowledge.embeddings.base import Embedder
from knowledge.embeddings.openai_compat import OpenAICompatibleEmbedder


class EmbeddingConfig(BaseModel):
    """Settings for the embedding model backend."""

    provider: str = "openai_compat"
    base_url: str = "http://localhost:8080"
    model: str | None = None
    dimension: int | None = None
    batch_size: int = 32
    timeout: float = 60.0
    retries: int = 3
    normalize: bool = True


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Construct the embedding implementation selected by config."""
    if cfg.provider == "openai_compat":
        return OpenAICompatibleEmbedder(
            base_url=cfg.base_url,
            model=cfg.model,
            dimension=cfg.dimension,
            batch_size=cfg.batch_size,
            timeout=cfg.timeout,
            retries=cfg.retries,
            normalize=cfg.normalize,
        )
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
