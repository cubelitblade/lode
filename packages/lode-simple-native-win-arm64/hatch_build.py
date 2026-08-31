"""Build hook that tags the wheel as platform-specific.

This distribution always ships a Windows arm64 binary, so the wheel tag is
static. Without this hook hatchling would emit a ``py3-none-any`` wheel, which
PyPI rejects for packages containing binaries.
"""

from __future__ import annotations

from typing import Any

from hatchling.builders.config import BuilderConfigBound
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PlatformTagBuildHook(BuildHookInterface[BuilderConfigBound]):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["tag"] = "py3-none-win_arm64"
