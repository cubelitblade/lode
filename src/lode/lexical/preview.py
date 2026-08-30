"""Token-stream preview: how a tokenizer actually splits a piece of text.

The FTS5 index is the only authority on what a tokenizer produces — the
strategy objects only carry configuration (the ``tokenize=`` clause), not a
Python-side reimplementation of the splitting rules. So instead of mirroring
each tokenizer in Python, this module asks SQLite itself: build a throwaway
in-memory FTS5 table with the same ``tokenize=`` clause, insert the text, and
read the token stream back through an ``fts5vocab`` *instance* table, which
exposes one row per token occurrence ordered by position.

The result is exactly what the index side stores for real chunks — including
normalization such as case folding and, for the native strategies, the Han
character + pinyin expansion. No index database is touched: everything happens
in a private ``:memory:`` connection.
"""

from __future__ import annotations

import sqlite3

from lode.lexical.base import LexicalStrategy
from lode.lexical.errors import ExtensionLoadError, detect_extension_capability


def tokenize_text(strategy: LexicalStrategy, text: str) -> list[str]:
    """Split ``text`` the way ``strategy``'s tokenizer would at index time.

    Returns the token stream in document order (one entry per occurrence,
    duplicates included). The strategy's ``setup`` runs on the throwaway
    connection so native tokenizers load their extension; the caller's
    database is never involved.
    """
    conn = sqlite3.connect(":memory:")
    try:
        # Only the native strategies load a shared extension; the built-in
        # FTS5 tokenizers need no extension loading at all, so on interpreters
        # without extension support the preview still works for them.
        if strategy.uses_helper:
            if not hasattr(conn, "enable_load_extension"):
                raise ExtensionLoadError(
                    detect_extension_capability(conn),
                    detail=f"the {strategy.name} tokenizer needs its native extension",
                )
            try:
                conn.enable_load_extension(True)
                strategy.setup(conn)
                conn.enable_load_extension(False)
            except sqlite3.Error as exc:
                raise ExtensionLoadError(
                    detect_extension_capability(conn),
                    detail=f"the {strategy.name} tokenizer failed to load: {exc}",
                ) from exc
        conn.execute(f"CREATE VIRTUAL TABLE preview USING fts5(text, tokenize='{strategy.tokenize_clause}')")
        conn.execute("INSERT INTO preview(rowid, text) VALUES (1, ?)", (text,))
        conn.execute("CREATE VIRTUAL TABLE preview_tokens USING fts5vocab('preview', 'instance')")
        rows = conn.execute("SELECT term FROM preview_tokens WHERE doc = 1 ORDER BY offset").fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()
