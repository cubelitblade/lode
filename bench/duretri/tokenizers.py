"""Tokenizer dimension for the benchmark.

Thin wrapper over ``lode.lexical``: the strategies live in the library, and
this module only adapts them to the benchmark's on-disk index building (which
calls ``setup`` then creates the FTS5 table with the strategy's clause).
"""

from __future__ import annotations

import sqlite3

from lode.lexical import STRATEGIES

# The tokenizers under test, keyed by name. ``jieba`` is a query-side strategy
# (it reuses the ``simple`` index), so it is excluded from the index dimension.
TOKENIZERS = {name: s for name, s in STRATEGIES.items() if name != "jieba"}


def create_table(conn: sqlite3.Connection, strategy_name: str) -> None:
    """Create the FTS5 table ``t`` for a strategy on ``conn``.

    The strategy contributes its tokenizer setup and ``tokenize=`` clause; the
    table creation itself stays here (mirroring how the schema layer owns DDL).
    """
    strategy = STRATEGIES[strategy_name]
    strategy.setup(conn)
    conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(text, tokenize='{strategy.tokenize_clause}')")
