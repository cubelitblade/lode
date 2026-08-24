"""Tests for the recursive text splitter.

Splitter behavior is deterministic and pure: these tests lock in chunk
sequence, size constraints, overlap semantics, and content addressing.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from lode.ingestion.digest import chunk_digest
from lode.ingestion.split import RecursiveTextSplitter


def _tail_overlap(prev: str, curr: str) -> int:
    """Longest k such that ``prev[-k:] == curr[:k]``."""
    for k in range(min(len(prev), len(curr)), 0, -1):
        if prev[-k:] == curr[:k]:
            return k
    return 0


def test_empty_text_yields_no_chunks() -> None:
    assert RecursiveTextSplitter().split("") == []


def test_short_text_is_single_chunk() -> None:
    chunks = RecursiveTextSplitter(chunk_size=100).split("short text")
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].seq == 0


def test_paragraphs_split_into_chunks() -> None:
    splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=10)
    text = "first paragraph content\n\nsecond paragraph content"
    chunks = splitter.split(text)
    assert len(chunks) == 2
    assert chunks[0].text == "first paragraph content\n\n"
    assert chunks[1].text == "second paragraph content"
    # Sequenced from 0 in file order.
    assert [c.seq for c in chunks] == [0, 1]


def test_chunks_respect_size_limit() -> None:
    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
    text = "word " * 100  # 500 chars
    chunks = splitter.split(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 50


def test_overlap_joins_consecutive_chunks() -> None:
    # Distinct words, so windows cannot line up with a period the way a
    # repeated word would. Each chunk must start with a non-empty tail of
    # the previous one (piece-level overlap, up to chunk_overlap).
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 3
    )
    splitter = RecursiveTextSplitter(chunk_size=60, chunk_overlap=15)
    chunks = splitter.split(words)
    assert len(chunks) > 1
    for prev, curr in pairwise(chunks):
        assert 0 < _tail_overlap(prev.text, curr.text) <= 15


def test_chunk_ids_are_content_addressed() -> None:
    splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=5)
    text = "one chunk\n\ntwo chunk"
    chunks = splitter.split(text)
    for chunk in chunks:
        assert chunk.digest == chunk_digest(chunk.text)
        assert chunk.digest.startswith("blake3:")


def test_stable_input_yields_stable_chunks() -> None:
    splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=5)
    text = "alpha beta gamma\n\ndelta epsilon zeta"
    assert splitter.split(text) == splitter.split(text)


def test_reordering_keeps_chunk_ids() -> None:
    # Moving a paragraph around must not change its chunk id: the id is
    # derived from the chunk text alone, not its position.
    splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=5)
    a = splitter.split("para one\n\npara two")
    b = splitter.split("para two\n\npara one")
    assert {c.digest for c in a} == {c.digest for c in b}


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=0)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)
