"""Tests for loading the bundled ``simple`` native extension."""

from __future__ import annotations

import os

from lode_simple_native import library_path


def test_library_path_exists_for_current_platform() -> None:
    """The platform binary selected by pip must resolve on disk.

    Guards against a platform distribution shipping without its binary (e.g.
    Windows reporting ``AMD64`` instead of ``x86_64`` in the marker).
    """
    path = library_path()
    assert os.path.isfile(path), f"bundled library missing at {path}"
