"""Tests for text extraction (formats, encodings, fallbacks)."""

from __future__ import annotations

from lode.ingestion.extract import decode_text, extract_text, is_supported


def test_extracts_utf8_plain_text() -> None:
    assert extract_text("héllo".encode(), ".txt") == "héllo"


def test_extracts_markdown() -> None:
    assert extract_text(b"# Title\n\nbody", ".md") == "# Title\n\nbody"


def test_strips_utf8_bom() -> None:
    assert extract_text("\ufeffhello".encode("utf-8"), ".txt") == "hello"


def test_falls_back_to_utf16() -> None:
    assert extract_text("héllo".encode("utf-16"), ".txt") == "héllo"


def test_latin1_never_fails() -> None:
    assert extract_text(b"caf\xe9", ".txt") == "café"


def test_unsupported_extension_returns_none() -> None:
    assert extract_text(b"binary", ".pdf") is None
    assert extract_text(b"x", ".docx") is None


def test_extension_matching_is_case_insensitive() -> None:
    assert extract_text(b"x", ".TXT") == "x"


def test_is_supported() -> None:
    assert is_supported(".txt")
    assert is_supported(".md")
    assert is_supported(".markdown")
    assert not is_supported(".docx")
    assert not is_supported("")


def test_decode_text_empty() -> None:
    assert decode_text(b"") == ""
