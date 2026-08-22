"""Text extraction: file bytes + suffix -> structured or plain text.

The extractor owns the supported-format table. Unsupported files return
None and are skipped by the pipeline. Plain formats (txt/md) decode to a
single unstructured segment; docx is parsed into structured segments that
carry a heading chain (the ``Title`` style as root, plus Word ``Heading N``
styles), which the chunker propagates to chunks so retrieval can cite
provenance (small-to-big).

Decoding is best-effort with a safe fallback chain: UTF-8 (with BOM) first,
then UTF-16, then Latin-1 — which never fails, so extraction always yields
some text rather than raising on an exotic encoding. (docx is binary, so it
is never routed through the free-text decoder.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

# Plain-text formats decode directly; docx is parsed structurally.
PLAIN_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
SUPPORTED_EXTENSIONS = PLAIN_EXTENSIONS | {".docx"}

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
    per heading section, each tagged with its heading chain.
    """
    suffix = suffix.lower()
    if suffix in PLAIN_EXTENSIONS:
        return [Segment(text=decode_text(data))]
    if suffix == ".docx":
        return _extract_docx(data)
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
