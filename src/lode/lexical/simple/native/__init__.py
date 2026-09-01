"""Loading the ``simple`` native extension.

The ``simple`` FTS5 tokenizer is a C shared library plus a jieba dictionary.
The shared library ships in the platform-specific ``lode-simple-native``
distribution selected by pip via wheel tags; the jieba dictionary ships here
in ``lode.lexical.simple.native``. ``load_simple`` loads the platform binary
(via ``lode_simple_native``) and points jieba at the dictionary.
"""

from __future__ import annotations

import os
import sqlite3
from importlib import resources

from lode_simple_native import library_path

from lode.lexical.errors import ExtensionLoadError, detect_extension_capability

#: Name of the jieba dictionary directory inside this package.
_DICT_DIR = "dict"


def _resource_path(name: str) -> str:
    """Return the on-disk path of a packaged resource, materializing if needed."""
    return str(resources.files(__package__).joinpath(name))


def load_simple(conn: sqlite3.Connection) -> None:
    """Load the ``simple`` extension and point jieba at its dictionary."""
    if not hasattr(conn, "enable_load_extension"):
        raise ExtensionLoadError(
            detect_extension_capability(conn),
            detail="the `simple` tokenizer needs its native extension",
        )
    try:
        conn.enable_load_extension(True)
        # ``sqlite3.Connection.load_extension`` requires a ``str``, not a
        # ``Path``, so convert via ``os.fspath``.
        conn.load_extension(os.fspath(library_path()))
        conn.execute(f"select jieba_dict('{_resource_path(_DICT_DIR)}')")
    except sqlite3.Error as exc:
        raise ExtensionLoadError(
            detect_extension_capability(conn),
            detail=f"the `simple` extension failed to load: {exc}",
        ) from exc
