"""Text extraction: file bytes + suffix -> structured or plain text.

The extractor owns the supported-format table. Unsupported files return
None and are skipped by the pipeline. Plain formats (txt/md) decode to a
single unstructured segment; docx is parsed into structured segments that
carry a heading chain (the ``Title`` style as root, plus Word ``Heading N``
styles); pdf is parsed into page-level segments that carry a page number
and, when the document has an outline, a heading chain mapped from it.
The chunker propagates heading/page so retrieval can cite provenance
(small-to-big).

Decoding is best-effort with a safe fallback chain: UTF-8 (with BOM) first,
then UTF-16, then Latin-1 — which never fails, so extraction always yields
some text rather than raising on an exotic encoding. (docx and pdf are
binary, so they are never routed through the free-text decoder.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from lode.ingestion.errors import ExtractionError

# Plain-text formats decode directly; docx and pdf are parsed structurally.
PLAIN_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
SUPPORTED_EXTENSIONS = PLAIN_EXTENSIONS | {".docx", ".pdf"}

# Decoding fallback chain, most specific first. UTF-16 is only attempted
# when a BOM is present: Python's utf-16 codec happily decodes arbitrary
# even-length bytes as garbage (native byte order), which would swallow
# Latin-1 text intended for the last fallback.
_DECODINGS = ("utf-8-sig", "utf-16", "latin-1")

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")

# Word built-in heading styles: "Heading 1".."Heading 9", plus the root
# "Title" style. Detection is conservative — only these built-ins are
# honoured (by name or style_id), never font-size/bold heuristics.
_HEADING_RE = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)
_TITLE_STYLE = "title"

# Separator between heading levels in a provenance chain.
_HEADING_SEP = " / "


@dataclass(frozen=True, slots=True)
class Segment:
    """A structural unit of extracted content carrying its provenance.

    A segment is the pre-chunk abstraction: free text plus the heading chain
    that identifies where it came from. The chunker turns a segment's text
    into one or more ``Chunk``s and propagates ``heading``/``page``.
    """

    text: str
    heading: str = ""
    page: int | None = None


def is_supported(suffix: str) -> bool:
    """Whether the file extension is extractable."""
    return suffix.lower() in SUPPORTED_EXTENSIONS


def extract_document(data: bytes, suffix: str) -> list[Segment] | None:
    """Structured extraction: a list of segments, or None for unsupported formats.

    Plain formats yield a single segment carrying the whole text (no heading),
    so the recursive chunker's behaviour is unchanged. docx yields one segment
    per heading section, each tagged with its heading chain. pdf yields one
    segment per page, tagged with the page number and (when the document has
    an outline) the heading chain in effect on that page.
    """
    suffix = suffix.lower()
    if suffix in PLAIN_EXTENSIONS:
        return [Segment(text=decode_text(data))]
    if suffix in (".docx", ".pdf"):
        # File bytes are untrusted input and the third-party parsers raise a
        # wide, unbounded set of exception types on malformed documents. This
        # is the one place a broad catch is correct: it converges that surface
        # to a single domain error the pipeline can treat as a per-file,
        # recoverable failure. Plain-text decoding never raises, so it is not
        # routed through here.
        try:
            if suffix == ".docx":
                return _extract_docx(data)
            return _extract_pdf(data)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"could not parse {suffix} document: {exc}") from exc
    return None


def extract_text(data: bytes, suffix: str) -> str | None:
    """Plain text for a supported file, or None for unsupported formats.

    For structured formats this flattens the segments (joining them). It is a
    convenience for callers that only need free text; the pipeline prefers
    ``extract_document``.
    """
    segments = extract_document(data, suffix)
    if segments is None:
        return None
    return "\n\n".join(segment.text for segment in segments)


def decode_text(data: bytes) -> str:
    """Decode bytes to text, falling back through encodings that cannot fail."""
    for encoding in _DECODINGS:
        if encoding == "utf-16" and not data.startswith(_UTF16_BOMS):
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 never raises; unreachable in practice, kept for exhaustiveness.
    return data.decode("latin-1", errors="replace")


def _extract_pdf(data: bytes) -> list[Segment]:
    """Parse a pdf into page-level segments.

    Each page becomes one segment carrying its 1-based page number. When the
    document has an outline (bookmarks), the heading chain in effect on each
    page is derived from it: outline entries are applied to the page they
    start on, and stay in effect until the next entry. Pages with no text are
    skipped, but page numbers keep their real values so citations stay
    accurate. No layout reconstruction, no block/paragraph extraction — the
    page is the unit, and finer structure is left to a future need.
    """
    document = pymupdf.open(stream=data, filetype="pdf")
    try:
        # Outline entries: [level, title, page], level 1-based, page 1-based.
        # PyMuPDF's stubs are partially unknown; cast to the documented shape.
        toc: list[tuple[int, str, int]] = document.get_toc(simple=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        outline = toc
        # Map each page to the heading chain in effect on it. Outline levels
        # are 1-based; the chain stack mirrors docx (level 1 resets to root).
        chain_by_page: dict[int, str] = {}
        stack: list[str] = []
        for level, title, page in outline:
            stack = [title] if level == 1 else [*stack[: level - 1], title]
            chain_by_page[page] = _HEADING_SEP.join(stack)

        segments: list[Segment] = []
        page_count = cast(int, document.page_count)  # pyright: ignore[reportUnknownMemberType]
        for page_index in range(page_count):
            page = document[page_index]
            text = cast(str, page.get_text("text")).strip()  # pyright: ignore[reportUnknownMemberType]
            if not text:
                continue
            page_number = page_index + 1
            segments.append(
                Segment(
                    text=text,
                    heading=chain_by_page.get(page_number, ""),
                    page=page_number,
                )
            )
        return segments
    finally:
        document.close()


def _extract_docx(data: bytes) -> list[Segment]:
    """Parse a docx into segments, one per heading section.

    Body paragraphs and tables under a heading are merged into a single
    segment so the chunker can window the whole section; heading boundaries
    are thus hard chunk boundaries. Paragraphs with no heading style (and no
    preceding heading) form a leading segment with an empty chain.
    """
    document = Document(BytesIO(data))
    segments: list[Segment] = []
    stack: list[str] = []  # heading chain text, e.g. ["报告", "第三章"]
    current_text: list[str] = []
    current_heading = ""

    def flush() -> None:
        nonlocal current_text
        if current_text:
            segments.append(Segment(text="\n".join(current_text), heading=current_heading))
            current_text = []

    # Iterate body children in document order (paragraphs interleaved with
    # tables), wrapping raw XML elements so we get the python-docx objects.
    # `document.element` is an untyped lxml tree; cast to `Any` so the strict
    # type checker does not flag the raw element access (scoped, not file-wide).
    element = cast(Any, document.element)
    for child in element.body:
        if child.tag == qn("w:p"):
            block = Paragraph(child, document)
            text = block.text.strip()
            if not text:
                continue
            level = _heading_level(block)
            if level is not None:
                flush()
                # Title resets the chain to the root; nested headings keep
                # parent levels 0..level-1, then push this heading.
                stack = [text] if level == 0 else [*stack[:level], text]
                current_heading = _HEADING_SEP.join(stack)
                current_text = [text]  # include the heading text in the section
            else:
                current_text.append(text)
        elif child.tag == qn("w:tbl"):
            block = Table(child, document)
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                current_text.append(" | ".join(cells))
    flush()
    return segments


def _heading_level(paragraph: Paragraph) -> int | None:
    """Return 0 for ``Title``, N for ``Heading N``, or None for body text.

    Conservative: match only the built-in style names and their style IDs
    (e.g. ``Heading1``), so localized or custom style names are treated as
    body text rather than being guessed at. A ``None`` style is body text.
    """
    style = paragraph.style
    if style is None:
        return None
    name = (style.name or "").strip().lower()
    style_id = (getattr(style, "style_id", "") or "").strip().lower()
    for candidate in (name, style_id):
        if candidate == _TITLE_STYLE:
            return 0
        match = _HEADING_RE.match(candidate)
        if match:
            return int(match.group(1))
    return None
