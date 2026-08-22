"""Tests for text extraction (formats, encodings, fallbacks)."""

from __future__ import annotations

import pymupdf

from lode.ingestion.extract import decode_text, extract_document, extract_text, is_supported


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
    assert extract_text(b"binary", ".exe") is None


def test_extension_matching_is_case_insensitive() -> None:
    assert extract_text(b"x", ".TXT") == "x"


def test_is_supported() -> None:
    assert is_supported(".txt")
    assert is_supported(".md")
    assert is_supported(".markdown")
    assert is_supported(".docx")
    assert is_supported(".pdf")
    assert not is_supported("")


def test_extract_document_plain_is_single_segment() -> None:
    segments = extract_document(b"hello world", ".txt")
    assert segments is not None
    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert segments[0].heading == ""
    assert segments[0].page is None


def test_decode_text_empty() -> None:
    assert decode_text(b"") == ""


def make_pdf_bytes() -> bytes:
    """A small pdf with an outline and a blank page, as raw bytes."""
    doc = pymupdf.open()
    doc.new_page()  # page 1
    doc[0].insert_text((72, 72), "第一章 第一节 内容", fontname="china-s")  # pyright: ignore[reportUnknownMemberType]
    doc.new_page()  # page 2: blank, skipped but keeps page numbering
    doc.new_page()  # page 3
    doc[2].insert_text((72, 72), "第二章 内容", fontname="china-s")  # pyright: ignore[reportUnknownMemberType]
    doc.set_toc(  # pyright: ignore[reportUnknownMemberType]
        [
            [1, "第一章", 1],
            [2, "第一节", 1],
            [1, "第二章", 3],
        ]
    )
    data = doc.tobytes()  # pyright: ignore[reportUnknownMemberType]
    doc.close()
    return data


def test_extract_pdf_page_level_segments() -> None:
    segments = extract_document(make_pdf_bytes(), ".pdf")
    assert segments is not None
    # Page 2 is blank and skipped; page numbers stay real (1 and 3).
    assert [segment.page for segment in segments] == [1, 3]
    assert "第一章 第一节 内容" in segments[0].text
    assert "第二章 内容" in segments[1].text


def test_extract_pdf_outline_maps_heading_chain() -> None:
    segments = extract_document(make_pdf_bytes(), ".pdf")
    assert segments is not None
    # Page 1 carries the deepest chain in effect; page 3 resets to the root.
    assert segments[0].heading == "第一章 / 第一节"
    assert segments[1].heading == "第二章"


def test_extract_pdf_without_outline_has_no_heading() -> None:
    doc = pymupdf.open()
    doc.new_page()
    doc[0].insert_text((72, 72), "plain pdf content")  # pyright: ignore[reportUnknownMemberType]
    data = doc.tobytes()  # pyright: ignore[reportUnknownMemberType]
    doc.close()

    segments = extract_document(data, ".pdf")
    assert segments is not None
    assert len(segments) == 1
    assert segments[0].page == 1
    assert segments[0].heading == ""
    assert "plain pdf content" in segments[0].text
