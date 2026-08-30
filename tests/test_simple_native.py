"""Tests for loading the bundled ``simple`` native extension."""

from __future__ import annotations

import os

# The helper is private by design; testing it directly is the point here.
from lode.lexical.simple.native import _library_path  # pyright: ignore[reportPrivateUsage]


def test_library_path_exists_for_current_platform() -> None:
    """The bundled binary for the running platform must resolve on disk.

    Guards against mapping drift between ``_LIBRARIES`` and the packaged
    ``native/`` layout (e.g. Windows reporting ``AMD64`` instead of ``x86_64``).
    """
    path = _library_path()
    assert os.path.isfile(path), f"bundled library missing at {path}"
