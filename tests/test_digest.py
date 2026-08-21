"""Tests for content-addressing primitives.

The normalization rule is a stability contract: changing it invalidates
every previously stored digest, so these tests lock it down.
"""

from __future__ import annotations

from lode.ingestion.digest import chunk_id, normalize


def test_normalize_strips_leading_and_trailing_whitespace() -> None:
    assert normalize("  hello  \n") == "hello"


def test_normalize_collapses_inline_whitespace_runs() -> None:
    assert normalize("a\t\tb  c\v\fd") == "a b c d"


def test_normalize_keeps_internal_newlines() -> None:
    # Newlines are structural: paragraph breaks must survive normalization.
    assert normalize("line one\n\nline two") == "line one\n\nline two"


def test_normalize_keeps_case_and_punctuation() -> None:
    assert normalize("Hello, World!") == "Hello, World!"


def test_normalize_empty_string() -> None:
    assert normalize("") == ""
    assert normalize("   \n\t  ") == ""


def test_chunk_id_is_prefixed_blake3() -> None:
    assert chunk_id("hello").startswith("blake3:")


def test_chunk_id_is_stable_for_same_text() -> None:
    assert chunk_id("hello world") == chunk_id("hello world")


def test_chunk_id_differs_for_different_text() -> None:
    assert chunk_id("hello") != chunk_id("world")


def test_chunk_id_ignores_insignificant_whitespace() -> None:
    # Content addressing is over the normalized form: whitespace-only edits
    # do not change the address (and thus reuse the embedding).
    assert chunk_id("a  b") == chunk_id("a b")
    assert chunk_id("  hello  ") == chunk_id("hello")


def test_chunk_id_keeps_newlines_distinct() -> None:
    # Newlines are meaningful: "a b" and "a\nb" address differently.
    assert chunk_id("a b") != chunk_id("a\nb")


def test_chunk_id_length() -> None:
    # blake3:<64 hex chars>
    assert len(chunk_id("x")) == len("blake3:") + 64


def test_normalize_and_chunk_id_agree() -> None:
    # The id is derived from the normalized text, so pre-normalizing is a no-op.
    assert chunk_id(normalize("  messy \t text  ")) == chunk_id("messy text")


def test_normalize_is_reversible_in_place() -> None:
    # Normalizing twice yields the same result (idempotent).
    assert normalize(normalize("  a \t b  ")) == normalize("  a \t b  ")
