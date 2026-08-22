"""Ingestion pipeline: file discovery, extraction, and chunking."""

from lode.ingestion.digest import chunk_id, normalize
from lode.ingestion.extract import Segment, extract_document
from lode.ingestion.split import Chunk, RecursiveSegmentSplitter, RecursiveTextSplitter, SegmentSplitter, Splitter

__all__ = [
    "Chunk",
    "RecursiveSegmentSplitter",
    "RecursiveTextSplitter",
    "Segment",
    "SegmentSplitter",
    "Splitter",
    "chunk_id",
    "extract_document",
    "normalize",
]
