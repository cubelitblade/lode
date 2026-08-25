"""Vendored text splitters, taken from ``langchain-text-splitters``.

This package keeps the upstream splitting behavior compatible while
removing the ``langchain-core``, tiktoken, and transformers dependencies
that made adopting the original package too expensive for lode.

See the ``base`` and ``character`` module docstrings for exact provenance
(pinned upstream commit SHAs) and the list of removed surface. License:
MIT, Copyright (c) LangChain, Inc.
"""

from lode._vendor.langchain_text_splitters.base import Language, TextSplitter
from lode._vendor.langchain_text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

__all__ = [
    "CharacterTextSplitter",
    "Language",
    "RecursiveCharacterTextSplitter",
    "TextSplitter",
]
