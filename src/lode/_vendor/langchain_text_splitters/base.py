"""Text splitter base class, vendored from ``langchain-text-splitters``.

Copyright (c) LangChain, Inc. Licensed under the MIT License.

Source: ``langchain_text_splitters.base`` in the langchain repository

Pinned to upstream commit ``fd822b07c00eefbe395ba3c285a4911f72a463de``
(2026-07-06, "fix(text-splitters): restore lazy imports for heavy optional
dependencies (#35469)"):
<https://github.com/langchain-ai/langchain/blob/fd822b07c00eefbe395ba3c285a4911f72a463de/libs/text-splitters/langchain_text_splitters/base.py>.

Vendored and reduced to the text-level core used by lode:

* kept: ``TextSplitter`` (constructor, validation, ``split_text``,
  ``_join_docs``, ``_merge_splits``) and the ``Language`` enum.
* removed: ``langchain-core`` integration (``Document``,
  ``BaseDocumentTransformer``, ``create_documents``/``split_documents``/
  ``transform_documents``), tokenizer support (tiktoken / transformers,
  ``TokenTextSplitter``, ``split_text_on_tokens``), and the
  ``add_start_index`` option (only used by the document methods).

The text splitting behavior is kept compatible with upstream.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class Language(str, Enum):  # noqa: UP042 — faithful upstream copy
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    R = "r"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"
    HASKELL = "haskell"
    ELIXIR = "elixir"
    POWERSHELL = "powershell"
    VISUALBASIC6 = "visualbasic6"


class TextSplitter(ABC):
    """Interface for splitting text into chunks."""

    # Class-level annotation beyond upstream: pyright narrows instance
    # attributes assigned in a ``__init__`` reached through ``**kwargs`` to
    # ``bool | str``, losing the literal. This keeps ``Literal["start", "end"]``.
    _keep_separator: bool | Literal["start", "end"]

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new `TextSplitter`.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                in each corresponding chunk `(True='start')`
            strip_whitespace: If `True`, strips whitespace from the start and end of
                every document

        Raises:
            ValueError: If `chunk_size` is less than or equal to 0
            ValueError: If `chunk_overlap` is less than 0
            ValueError: If `chunk_overlap` is greater than `chunk_size`
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            msg = f"Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller."
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            len_ = self._length_function(d)
            if total + len_ + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the specified %d",
                        total,
                        self._chunk_size,
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs
