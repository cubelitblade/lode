"""Chunk model and splitter interface.

A `Splitter` turns extracted text into a sequence of `Chunk`s. Chunk
identity is content-addressed via `chunk_id`, so the model and the
splitter interface must stay stable: downstream layers (index store,
incremental diff, search) all depend on this shape.

The recursive splitter delegates to a vendored copy of LangChain's
``RecursiveCharacterTextSplitter`` (see ``lode.ingestion.vendored``).
This module owns the lode contract on top of it: `Chunk` shape, content
addressing, separator priority, and input validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lode.ingestion.digest import chunk_id
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
        id: content address, ``blake3:<hex>`` of the normalized text.
        text: the raw (unnormalized) text; what gets embedded and returned.
        seq: position of this chunk within its file, 0-based.
        heading: heading chain of this chunk (anchor/citation), may be empty.
        page: source page number (PDFs), or None otherwise.
    """

    id: str
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
    def split(self, text: str) -> list[Chunk]:
        """Split text into chunks with sequential positions (0-based)."""


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
    shifts subsequent chunks, but content addressing makes that harmless:
    unchanged chunk text keeps its id and reuses its embedding (see PLAN D4).
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

    def split(self, text: str) -> list[Chunk]:
        pieces = self._splitter.split_text(text)
        return [Chunk(id=chunk_id(piece), text=piece, seq=seq) for seq, piece in enumerate(pieces)]


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
                        id=piece.id,
                        text=piece.text,
                        seq=seq,
                        heading=segment.heading,
                        page=segment.page,
                    )
                )
                seq += 1
        return chunks
