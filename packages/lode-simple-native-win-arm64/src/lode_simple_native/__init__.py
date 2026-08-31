"""Native ``simple`` FTS5 tokenizer binary for Windows arm64.

This distribution ships only the platform-specific shared library. The
``lode`` main package depends on the correct platform distribution via
environment markers, so importing this module always yields the binary for
the running platform.
"""

from __future__ import annotations

from importlib import resources

__all__ = ["library_path"]


def library_path() -> str:
    """Return the on-disk path of the bundled shared library.

    The path is materialized on disk (via ``importlib.resources``) so it can
    be passed to ``sqlite3.Connection.load_extension``.
    """
    return str(resources.files(__package__).joinpath("lib/simple.dll"))
