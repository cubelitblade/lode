"""Embedding backends: `Embedder` interface + concrete clients."""

from lode.embeddings.base import Embedder, l2_normalize
from lode.embeddings.ollama import OllamaEmbedder
from lode.embeddings.openai_compat import OpenAICompatibleEmbedder
from lode.embeddings.tei_native import HuggingFaceTEINativeEmbedder

__all__ = [
    "Embedder",
    "HuggingFaceTEINativeEmbedder",
    "OllamaEmbedder",
    "OpenAICompatibleEmbedder",
    "l2_normalize",
]
