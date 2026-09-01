"""Build hook that tags the wheel as platform-specific and bundles the binary.

This distribution ships a single platform binary selected at build time. The
wheel tag is derived from the build environment, and only the matching binary
under ``native/<platform>/`` is copied into the wheel as
``lode_simple_native/lib/<binary>``.

Without this hook hatchling would emit a ``py3-none-any`` wheel, which PyPI
rejects for packages containing binaries.

For editable installs (``uv sync`` / ``pip install -e``) the source tree keeps
binaries under ``native/<platform>/``, so the hook copies the current
platform's binary into ``src/lode_simple_native/lib/`` to keep the runtime
layout identical to the wheel (``lode_simple_native/lib/<binary>``).
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Maps (sys.platform, platform.machine()) to (wheel tag suffix, native dir, binary name).
#: ``platform.machine()`` returns ``AMD64``/``ARM64`` on Windows but
#: ``x86_64``/``aarch64`` on Linux/macOS, so the keys are platform-specific.
#: The Linux tag is ``manylinux_2_34`` because the binary links against
#: ``GLIBC_2.32`` / ``GLIBCXX_3.4.29`` (verified via ``auditwheel show``); a
#: lower tag would let pip install it on older glibc systems where it fails to
#: load.
_PLATFORMS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("linux", "x86_64"): ("manylinux_2_34_x86_64", "linux-x86_64", "libsimple.so"),
    ("linux", "aarch64"): ("manylinux_2_34_aarch64", "linux-aarch64", "libsimple.so"),
    ("darwin", "arm64"): ("macosx_11_0_arm64", "macos-arm64", "libsimple.dylib"),
    ("darwin", "x86_64"): ("macosx_10_9_x86_64", "macos-x86_64", "libsimple.dylib"),
    ("win32", "AMD64"): ("win_amd64", "windows-amd64", "simple.dll"),
    ("win32", "ARM64"): ("win_arm64", "windows-arm64", "simple.dll"),
}


class PlatformTagBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        key = (sys.platform, platform.machine())
        try:
            tag_suffix, native_dir, binary = _PLATFORMS[key]
        except KeyError as exc:
            raise SystemExit(f"unsupported platform: {key}") from exc

        if version == "editable":
            # Editable installs map the source tree directly, so the wheel's
            # ``force_include`` never runs. Copy the current platform binary
            # into the source ``lib/`` so ``library_path()`` resolves it.
            root = Path(self.root)
            src = root / "native" / native_dir / binary
            dst = root / "src" / "lode_simple_native" / "lib" / binary
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return

        build_data["tag"] = f"py3-none-{tag_suffix}"
        build_data["force_include"] = {
            f"native/{native_dir}/{binary}": f"lode_simple_native/lib/{binary}",
        }
