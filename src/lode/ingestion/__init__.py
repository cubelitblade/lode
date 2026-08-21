"""Ingestion pipeline: file discovery, extraction, and chunking."""

from lode.ingestion.digest import chunk_id, normalize
from lode.ingestion.split import Chunk, RecursiveTextSplitter, Splitter

__all__ = [
    "Chunk",
    "RecursiveTextSplitter",
    "Splitter",
    "chunk_id",
    "normalize",
]
