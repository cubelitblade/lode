"""The ``simple`` lexical strategy.

``simple`` is a native FTS5 tokenizer that indexes each Han character plus its
pinyin, and offers ``simple_query`` / ``jieba_query`` helpers for query
construction. The strategy and its bundled native library live together in
this package (``lode.lexical.simple``); the library is under ``native/``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from lode.lexical.base import IndexedTerm, identity_terms


@dataclass(frozen=True, slots=True)
class SimpleStrategy:
    """Native ``simple`` tokenizer: Han characters + pinyin.

    Requires the ``simple`` shared library (see ``lode.lexical.simple.native``).
    The query side uses the ``simple_query`` helper; ``jieba`` uses
    ``jieba_query`` for word-level matching.
    """

    name: str
    tokenize_clause: str = "simple"
    #: Which native helper the query side runs through.
    helper: str = "simple_query"
    uses_helper: bool = True

    def setup(self, conn: sqlite3.Connection) -> None:
        from lode.lexical.simple.native import load_simple

        load_simple(conn)

    def query(self, text: str) -> str:
        # The raw query is passed as a bound parameter to the helper; the
        # return value here is unused when ``uses_helper`` is true.
        return text

    def interpret(self, tokens: Sequence[str]) -> list[IndexedTerm]:
        # Structuring the pinyin readings into variants needs a contract from
        # the native side (see discussion); until then the stream stays flat.
        return identity_terms(tokens)


__all__ = ["SimpleStrategy"]
