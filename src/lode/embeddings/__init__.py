"""Embedding backends: `Embedder` interface + OpenAI-compatible client."""

from lode.embeddings.base import Embedder, l2_normalize
from lode.embeddings.openai_compat import OpenAICompatibleEmbedder

__all__ = ["Embedder", "OpenAICompatibleEmbedder", "l2_normalize"]
