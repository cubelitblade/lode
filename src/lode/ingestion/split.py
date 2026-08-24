"""Chunk model and splitter interface.

A `Splitter` turns extracted text into a sequence of text pieces. The
interface is deliberately generic — text in, pieces out — so it stays
reusable across future splitter kinds (anchor-based, code-aware, ...).
Chunk identity (content addressing via `chunk_digest`) and provenance
(heading/page) are *not* the splitter's concern: they are assembled by
`RecursiveSegmentSplitter`, the single place that builds `Chunk`s.

The recursive splitter delegates to a vendored copy of LangChain's
``RecursiveCharacterTextSplitter`` (see ``lode.ingestion.vendored``).
This module owns the lode contract on top of it: the piece contract,
separator priority, and input validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lode.ingestion.digest import chunk_digest
from lode.ingestion.extract import Segment
from lode.ingestion.vendored import (
    RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter,
)

# Separator priority for recursive splitting: paragraph breaks first, then
# line breaks, then CJK sentence/pause marks, then word spaces, then
# character-level hard splits. Separators are kept attached to the piece
# before them so chunk text preserves structure (rendering + highlighting).
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # noqa: RUF001 — fullwidth CJK punctuation is intentional


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text.

    Attributes:
        digest: content address, ``blake3:<hex>`` of the normalized text.
        text: the raw (unnormalized) text; what gets embedded and returned.
        seq: position of this chunk within its file, 0-based.
        heading: heading chain of this chunk (anchor/citation), may be empty.
        page: source page number (PDFs), or None otherwise.
    """

    digest: str
    text: str
    seq: int
    heading: str = ""
    page: int | None = None


class Splitter(ABC):
    """Interface for chunking extracted text.

    Implementations are pure string algorithms: no I/O, no side effects,
    deterministic for a given input. Swap in anchor-based splitters
    (markdown headings, code functions) by replacing the implementation;
    the core pipeline does not change.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into text pieces (no provenance, no content addressing)."""


class RecursiveTextSplitter(Splitter):
    """Continuous-window recursive splitter with greedy merge.

    Wraps the vendored LangChain recursive splitter:

    * ``keep_separator="end"`` attaches separators (paragraph breaks, CJK
      sentence marks) to the piece before them, so chunk text preserves
      structure for rendering and highlighting.
    * ``strip_whitespace=False`` keeps that structure intact.
    * Input validation is stricter than upstream: ``chunk_overlap`` must be
      strictly smaller than ``chunk_size``.

    Deterministic for a given input. A change in the middle of a file
    shifts subsequent pieces, but content addressing (applied later by the
    segment splitter) makes that harmless: unchanged piece text keeps its
    digest and reuses its embedding.
    """

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self._splitter = _RecursiveCharacterTextSplitter(
            separators=_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator="end",
            strip_whitespace=False,
        )

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


class SegmentSplitter(ABC):
    """Interface for chunking structured extracted content.

    Operates on a sequence of ``Segment``s (which carry a heading chain and
    optional page). Implementations are pure string algorithms: no I/O, no
    side effects, deterministic for a given input. This is the entry point
    the pipeline uses for formats with structure (docx today, pdf/xlsx
    later); plain formats degrade to a single unstyled segment.
    """

    @abstractmethod
    def split_segments(self, segments: list[Segment]) -> list[Chunk]:
        """Split segments into chunks with global sequential positions (0-based)."""


class RecursiveSegmentSplitter(SegmentSplitter):
    """Heading-aware recursive splitter: window each segment independently.

    Each segment's text is windowed by the same recursive rules as
    ``RecursiveTextSplitter``, and the resulting chunks are tagged with the
    segment's ``heading``/``page``. Heading boundaries are hard chunk
    boundaries (no overlap across sections), which keeps provenance exact.

    A plain format arrives as a single unstyled segment, so the output is
    identical to windowing the whole text with ``RecursiveTextSplitter`` —
    only ``seq`` is re-numbered globally.
    """

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_segments(self, segments: list[Segment]) -> list[Chunk]:
        chunks: list[Chunk] = []
        seq = 0
        for segment in segments:
            for piece in self._splitter.split(segment.text):
                chunks.append(
                    Chunk(
                        digest=chunk_digest(piece),
                        text=piece,
                        seq=seq,
                        heading=segment.heading,
                        page=segment.page,
                    )
                )
                seq += 1
        return chunks
