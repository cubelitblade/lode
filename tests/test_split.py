"""Tests for the recursive text splitter.

Splitter behavior is deterministic and pure: these tests lock in piece
sequence, size constraints, and overlap semantics. The splitter returns
plain text pieces (``list[str]``); content addressing and provenance are
the segment splitter's job (see ``test_segment_split.py``).
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from lode.ingestion.split import RecursiveTextSplitter


def _tail_overlap(prev: str, curr: str) -> int:
    """Longest k such that ``prev[-k:] == curr[:k]``."""
    for k in range(min(len(prev), len(curr)), 0, -1):
        if prev[-k:] == curr[:k]:
            return k
    return 0


def test_empty_text_yields_no_pieces() -> None:
    assert RecursiveTextSplitter().split("") == []


def test_short_text_is_single_piece() -> None:
    pieces = RecursiveTextSplitter(chunk_size=100).split("short text")
    assert pieces == ["short text"]


def test_paragraphs_split_into_pieces() -> None:
    splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=10)
    text = "first paragraph content\n\nsecond paragraph content"
    pieces = splitter.split(text)
    assert pieces == ["first paragraph content\n\n", "second paragraph content"]


def test_pieces_respect_size_limit() -> None:
    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
    text = "word " * 100  # 500 chars
    pieces = splitter.split(text)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 50


def test_overlap_joins_consecutive_pieces() -> None:
    # Distinct words, so windows cannot line up with a period the way a
    # repeated word would. Each piece must start with a non-empty tail of
    # the previous one (piece-level overlap, up to chunk_overlap).
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 3
    )
    splitter = RecursiveTextSplitter(chunk_size=60, chunk_overlap=15)
    pieces = splitter.split(words)
    assert len(pieces) > 1
    for prev, curr in pairwise(pieces):
        assert 0 < _tail_overlap(prev, curr) <= 15


def test_stable_input_yields_stable_pieces() -> None:
    splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=5)
    text = "alpha beta gamma\n\ndelta epsilon zeta"
    assert splitter.split(text) == splitter.split(text)


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=0)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)
