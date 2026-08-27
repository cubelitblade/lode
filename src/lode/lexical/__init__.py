"""Lexical strategies for FTS5 tokenization and query construction.

A ``LexicalStrategy`` bundles how documents are indexed (the FTS5
``tokenize=`` clause plus any native setup) with how a user query is turned
into a MATCH expression. The built-in strategies are ``unicode61`` (SQLite
default), ``trigram`` (CJK-friendly 3-grams), and ``simple`` / ``jieba``
(native extension, when the shared library is available).
"""

from lode.lexical.base import IndexedTerm, LexicalStrategy, distinct_terms
from lode.lexical.preview import tokenize_text
from lode.lexical.simple import SimpleStrategy
from lode.lexical.strategies import (
    HELPER_SQL,
    STRATEGIES,
    TrigramStrategy,
    Unicode61Strategy,
)

__all__ = [
    "HELPER_SQL",
    "STRATEGIES",
    "IndexedTerm",
    "LexicalStrategy",
    "SimpleStrategy",
    "TrigramStrategy",
    "Unicode61Strategy",
    "distinct_terms",
    "tokenize_text",
]
