"""Native ``simple`` FTS5 tokenizer binary.

This distribution ships only the platform-specific shared library for the
running platform. The ``lode`` main package depends on this distribution, and
``library_path`` resolves the bundled binary for the current platform.
"""

from __future__ import annotations

import platform
import sys
from importlib import resources
from pathlib import Path

__all__ = ["library_path"]

#: Maps (sys.platform, platform.machine()) to (native subdir, binary name).
#: ``platform.machine()`` returns ``AMD64``/``ARM64`` on Windows but
#: ``x86_64``/``aarch64`` on Linux/macOS, so the keys are platform-specific.
_PLATFORMS: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): ("linux-x86_64", "libsimple.so"),
    ("linux", "aarch64"): ("linux-aarch64", "libsimple.so"),
    ("darwin", "arm64"): ("macos-arm64", "libsimple.dylib"),
    ("darwin", "x86_64"): ("macos-x86_64", "libsimple.dylib"),
    ("win32", "AMD64"): ("windows-amd64", "simple.dll"),
    ("win32", "ARM64"): ("windows-arm64", "simple.dll"),
}


def library_path() -> Path:
    """Return the on-disk path of the bundled shared library.

    The path is materialized on disk (via ``importlib.resources``) so it can
    be passed to ``sqlite3.Connection.load_extension``.

    The binary always lives at ``lode_simple_native/lib/<binary>``: the build
    hook copies it there for both wheel and editable installs.
    """
    key = (sys.platform, platform.machine())
    try:
        _, name = _PLATFORMS[key]
    except KeyError as exc:
        raise RuntimeError(f"unsupported platform: {key}") from exc

    return resources.files(__package__).joinpath("lib", name)
