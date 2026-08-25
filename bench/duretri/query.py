"""Query strategy dimension for the benchmark.

Thin wrapper over ``lode.lexical``. The benchmark keeps tokenizer and query
strategy as two independent axes so it can measure cross combinations, even
though the library bundles them into one ``LexicalStrategy``; here we expose
the query side (``query`` / ``uses_helper``) and the helper SQL separately.
"""

from __future__ import annotations

import sqlite3

from lode.lexical import HELPER_SQL, STRATEGIES

# The query strategies under test, keyed by name (re-exported from the library).
QUERY_STRATEGIES = STRATEGIES

# Recommended query strategy for each tokenizer.
RECOMMENDED_STRATEGY: dict[str, str] = {
    "unicode61": "unicode61",
    "trigram": "trigram",
    "simple": "simple",
}

# The helper SQL used for strategies backed by the ``simple`` extension.
SIMPLE_HELPER_SQL = HELPER_SQL


def match(conn: sqlite3.Connection, strategy_name: str, query: str) -> str:
    """Return the MATCH argument for ``query`` under a strategy.

    For strategies backed by a native helper, the raw query is returned (the
    caller passes it as a bound parameter to the helper); otherwise the
    strategy's ``query`` builds the MATCH expression directly.
    """
    strategy = STRATEGIES[strategy_name]
    if strategy.uses_helper:
        return query
    return strategy.query(query)
