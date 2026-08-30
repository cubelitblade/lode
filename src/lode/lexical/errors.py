"""Exceptions and capability detection for loading SQLite extensions.

The ``simple`` tokenizer (lexical layer) and the vec0 index (index layer)
both load shared SQLite extensions. Whether the running Python's ``sqlite3``
module can do so is a property of the interpreter build, not of lode:
``enable_load_extension`` and ``load_extension`` are gated by the same CPython
compile-time macro (``PY_SQLITE_ENABLE_LOAD_EXTENSION``). This module owns
that detection and the error raised when it fails, so every call site reports
the same clear, metadata-carrying failure instead of a bare ``AttributeError``.
"""

from __future__ import annotations

import platform
import sqlite3
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ExtensionCapability:
    """Whether this Python's ``sqlite3`` module can load shared extensions.

    ``can_load`` is the single capability signal (both methods are gated by
    the same CPython macro). The remaining fields explain *why*, so errors and
    test skips can carry the same metadata.
    """

    can_load: bool
    python: str
    sqlite_version: str
    omit_load_extension: bool

    def skip_reason(self) -> str:
        """A metadata-rich reason for skipping native-extension tests."""
        return (
            f"sqlite3 cannot load extensions: python={self.python}, "
            f"sqlite={self.sqlite_version}, "
            f"SQLITE_OMIT_LOAD_EXTENSION={'present' if self.omit_load_extension else 'absent'}"
        )


def detect_extension_capability(conn: sqlite3.Connection | None = None) -> ExtensionCapability:
    """Probe the running interpreter's extension-loading capability.

    Uses ``conn`` when given (the caller's own connection), else a throwaway
    in-memory one. The capability is a property of the interpreter build, so
    either way the result is the same.
    """
    close = conn is None
    if conn is None:
        conn = sqlite3.connect(":memory:")
    try:
        can_load = hasattr(conn, "enable_load_extension")
        options = {str(row[0]) for row in conn.execute("PRAGMA compile_options")}
        omit = "SQLITE_OMIT_LOAD_EXTENSION" in options
    finally:
        if close:
            conn.close()
    return ExtensionCapability(
        can_load=can_load,
        python=platform.python_version(),
        sqlite_version=sqlite3.sqlite_version,
        omit_load_extension=omit,
    )


class ExtensionLoadError(Exception):
    """The running Python's ``sqlite3`` module cannot load shared extensions.

    Raised where lode needs a native SQLite extension (sqlite-vec for the
    index, the ``simple`` tokenizer for lexical search) but the interpreter
    was built without ``--enable-loadable-sqlite-extensions``. Carries the
    capability metadata so callers can present a precise recovery message.
    """

    # Stable machine-readable identifier for the CLI/MCP error envelope.
    code: ClassVar[str] = "extension_load"

    def __init__(self, capability: ExtensionCapability, *, detail: str = "") -> None:
        self.capability = capability
        self.detail = detail
        super().__init__(self._diagnostic())

    def _diagnostic(self) -> str:
        cap = self.capability
        base = (
            f"this Python cannot load SQLite extensions "
            f"(python {cap.python}, sqlite {cap.sqlite_version}, "
            f"SQLITE_OMIT_LOAD_EXTENSION={'present' if cap.omit_load_extension else 'absent'})"
        )
        return f"{base}: {self.detail}" if self.detail else base

    def template_fields(self) -> dict[str, object]:
        cap = self.capability
        return {
            "python": cap.python,
            "sqlite_version": cap.sqlite_version,
            "omit_load_extension": "present" if cap.omit_load_extension else "absent",
            "detail": self.detail,
        }
