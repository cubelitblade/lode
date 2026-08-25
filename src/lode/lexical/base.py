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
from typing import Protocol


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

    @property
    def uses_helper(self) -> bool:
        """Whether ``query`` must be run through a native SQL helper.

        When true, the caller invokes the helper (``simple_query(?)`` /
        ``jieba_query(?)``) with the raw query as a bound parameter instead of
        using ``query``'s return value directly as the MATCH expression.
        """
        ...
