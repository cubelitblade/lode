"""Embedding backends: `Embedder` interface + OpenAI-compatible client."""

from knowledge.embeddings.base import Embedder, l2_normalize
from knowledge.embeddings.openai_compat import OpenAICompatibleEmbedder

__all__ = ["Embedder", "OpenAICompatibleEmbedder", "l2_normalize"]
