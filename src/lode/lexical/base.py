"""Lexical strategies: how text is tokenized and queried in FTS5.

A ``LexicalStrategy`` bundles the two sides of one lexical concept — how
documents are indexed (the FTS5 ``tokenize=`` clause plus any SQLite setup the
tokenizer needs) and how a user query is turned into an FTS5 MATCH expression.
The two must mirror each other: the query side has to split text the same way
the index side does, or matches never line up.

The strategy does *not* create the FTS5 table itself; that stays with the
schema layer, which calls ``tokenize_clause`` and ``setup`` and then issues the
``CREATE VIRTUAL TABLE``. Keeping table creation in one place means the schema
owns the full DDL and a strategy only contributes its tokenizer-specific bits.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class IndexedTerm:
    """One structured term of the index-side token stream.

    ``surface`` is the term's primary form as stored in the index (a word for
    ``unicode61``, a 3-gram for ``trigram``, a Han character for the native
    strategies). ``variants`` are accompanying forms of the same surface that
    the tokenizer also indexes — e.g. the pinyin readings emitted alongside
    each Han character. Empty for tokenizers that index one plain form.
    """

    surface: str
    variants: tuple[str, ...] = ()


def identity_terms(tokens: Sequence[str]) -> list[IndexedTerm]:
    """The default interpretation: every raw token is its own plain term."""
    return [IndexedTerm(surface=token) for token in tokens]


def distinct_terms(terms: Sequence[IndexedTerm]) -> list[IndexedTerm]:
    """Merge repeated surfaces, preserving first-seen order.

    Variants of repeated surfaces unite in first-seen order; everything else
    keeps the original occurrence order.
    """
    merged: dict[str, IndexedTerm] = {}
    order: list[str] = []
    for term in terms:
        if term.surface not in merged:
            merged[term.surface] = IndexedTerm(term.surface, term.variants)
            order.append(term.surface)
            continue
        seen = merged[term.surface]
        if extra := tuple(v for v in term.variants if v not in seen.variants):
            merged[term.surface] = IndexedTerm(seen.surface, (*seen.variants, *extra))
    return [merged[surface] for surface in order]


class LexicalStrategy(Protocol):
    """One lexical strategy: index configuration + query construction.

    Implementations are immutable and side-effect free to construct; any
    runtime setup (e.g. loading a native extension) happens in ``setup``.
    """

    @property
    def name(self) -> str:
        """Name used to select this strategy in configuration."""
        ...

    @property
    def tokenize_clause(self) -> str:
        """The FTS5 ``tokenize=`` clause, e.g. ``"trigram"`` or ``"simple"``."""
        ...

    def setup(self, conn: sqlite3.Connection) -> None:
        """Run any SQLite setup the tokenizer needs before the table exists.

        Called once per connection before ``CREATE VIRTUAL TABLE``. The default
        is a no-op; native tokenizers load their extension here.
        """
        ...

    def query(self, text: str) -> str:
        """Build the FTS5 MATCH expression for a user ``text``.

        The result is passed as the MATCH argument. Strategies backed by a
        native helper (e.g. ``simple_query``) return a marker that the caller
        resolves to the helper call; see ``uses_helper``.
        """
        ...

    def interpret(self, tokens: Sequence[str]) -> list[IndexedTerm]:
        """Structure a raw index-side token stream into terms.

        The adapter between what the tokenizer stores and what consumers
        (e.g. ``assay how``) display: each strategy knows its own token
        conventions and folds accompanying forms into ``variants``. The
        occurrence order of the stream is preserved; duplicates are kept
        (callers dedupe with :func:`distinct_terms` when displaying).
        """
        ...

    @property
    def uses_helper(self) -> bool:
        """Whether ``query`` must be run through a native SQL helper.

        When true, the caller invokes the helper (``simple_query(?)`` /
        ``jieba_query(?)``) with the raw query as a bound parameter instead of
        using ``query``'s return value directly as the MATCH expression.
        """
        ...
