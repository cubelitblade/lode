"""Loading the ``simple`` native extension.

The ``simple`` FTS5 tokenizer is a C shared library plus a jieba dictionary.
Both ship inside this package (``lode.lexical.simple.native``) so the extension
is self-contained. The shared library is platform-specific; ``load_simple``
picks the right one for the current OS/architecture via ``importlib.resources``
and loads it into a connection.
"""

from __future__ import annotations

import platform
import sqlite3
import sys
from importlib import resources

#: Name of the jieba dictionary directory inside this package.
_DICT_DIR = "dict"

#: Mapping of (platform, machine) -> (subdir, library filename).
#: ``machine`` values are normalized to the names used in the release assets.
_LIBRARIES: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): ("linux", "libsimple.so"),
    ("linux", "aarch64"): ("linux", "libsimple.so"),
    ("darwin", "arm64"): ("darwin/arm64", "libsimple.dylib"),
    ("darwin", "x86_64"): ("darwin/x86_64", "libsimple.dylib"),
    ("win32", "arm64"): ("windows/arm64", "simple.dll"),
    ("win32", "x86_64"): ("windows/x86_64", "simple.dll"),
}

#: ``platform.machine()`` spellings that all mean x86-64.
_X86_64_MACHINES = frozenset({"x86_64", "amd64", "x86-64"})


def _normalized_machine() -> str:
    """Return ``platform.machine()`` normalized to the release-asset naming.

    Windows reports ``AMD64`` for x86-64, which must map to ``x86_64``.
    """
    machine = platform.machine().lower()
    return "x86_64" if machine in _X86_64_MACHINES else machine


def _resource_path(name: str) -> str:
    """Return the on-disk path of a packaged resource, materializing if needed."""
    return str(resources.files(__package__).joinpath(name))


def _library_path() -> str:
    """Return the packaged library path for the current platform.

    Raises ``RuntimeError`` when no binary is bundled for this platform.
    """
    machine = _normalized_machine()
    key = (sys.platform, machine)
    entry = _LIBRARIES.get(key)
    if entry is None:
        raise RuntimeError(f"the `simple` tokenizer has no bundled binary for platform {sys.platform}/{machine}")
    subdir, filename = entry
    return _resource_path(f"{subdir}/{filename}")


def load_simple(conn: sqlite3.Connection) -> None:
    """Load the ``simple`` extension and point jieba at its dictionary."""
    conn.enable_load_extension(True)
    conn.load_extension(_library_path())
    conn.execute(f"select jieba_dict('{_resource_path(_DICT_DIR)}')")
