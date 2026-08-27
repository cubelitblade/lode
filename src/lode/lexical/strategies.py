"""Built-in lexical strategies.

``unicode61`` is SQLite's default tokenizer (word-ish tokens); ``trigram``
indexes 3-grams, which is far better for CJK substring matching. ``simple`` is
a native extension that indexes each Han character plus its pinyin and offers
``simple_query`` / ``jieba_query`` helpers; it lives in ``lode.lexical.simple``
and is only usable when the native library is present.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from lode.lexical.base import IndexedTerm, LexicalStrategy, identity_terms
from lode.lexical.simple import SimpleStrategy

# Word-ish tokens for the plain (unicode61-style) query. Each token is quoted
# so punctuation in user queries cannot break the MATCH syntax.
_WORD = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class Unicode61Strategy:
    """SQLite's default tokenizer: split on word characters."""

    name: str = "unicode61"
    tokenize_clause: str = "unicode61"
    uses_helper: bool = False

    def setup(self, conn: sqlite3.Connection) -> None:
        return None

    def query(self, text: str) -> str:
        tokens = _WORD.findall(text)
        return " OR ".join(f'"{token}"' for token in tokens)

    def interpret(self, tokens: Sequence[str]) -> list[IndexedTerm]:
        return identity_terms(tokens)


@dataclass(frozen=True, slots=True)
class TrigramStrategy:
    """Trigram tokenizer: index and query by 3-grams (CJK-friendly)."""

    name: str = "trigram"
    tokenize_clause: str = "trigram"
    uses_helper: bool = False

    def setup(self, conn: sqlite3.Connection) -> None:
        return None

    def query(self, text: str) -> str:
        grams = [text[i : i + 3] for i in range(len(text) - 2)]
        return " OR ".join(f'"{g}"' for g in grams)

    def interpret(self, tokens: Sequence[str]) -> list[IndexedTerm]:
        return identity_terms(tokens)


# The strategies under test, keyed by name.
STRATEGIES: dict[str, LexicalStrategy] = {
    "unicode61": Unicode61Strategy(),
    "trigram": TrigramStrategy(),
    "simple": SimpleStrategy("simple", helper="simple_query"),
    "jieba": SimpleStrategy("jieba", helper="jieba_query"),
}

# The helper SQL used for strategies backed by the ``simple`` extension.
HELPER_SQL: dict[str, str] = {
    "simple": "simple_query(?)",
    "jieba": "jieba_query(?)",
}
