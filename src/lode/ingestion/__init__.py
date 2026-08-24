"""Ingestion pipeline: file discovery, extraction, and chunking."""

from lode.ingestion.digest import chunk_digest, normalize
from lode.ingestion.extract import Segment, extract_document
from lode.ingestion.split import Chunk, RecursiveSegmentSplitter, RecursiveTextSplitter, SegmentSplitter, Splitter

__all__ = [
    "Chunk",
    "RecursiveSegmentSplitter",
    "RecursiveTextSplitter",
    "Segment",
    "SegmentSplitter",
    "Splitter",
    "chunk_digest",
    "extract_document",
    "normalize",
]
