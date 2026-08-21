"""File discovery: walk a workspace and apply ignore rules.

Discovery is format-agnostic — it returns every file (as a
workspace-relative path) that survives the ignore rules. Whether a file can
actually be ingested is the extractor's decision, keeping the two concerns
separate.

Ignore matching uses glob patterns against the POSIX form of the relative
path (``.git/**``, ``*.tmp``, ``.lode/**``). A bare directory pattern
(e.g. ``.git``) also ignores everything under it, mirroring .gitignore's
directory semantics.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from pathlib import Path

# Always ignored: the runtime data directory holding the index store.
DEFAULT_IGNORES = (".lode/**",)


def discover(root: Path, patterns: Sequence[str] = ()) -> list[Path]:
    """Workspace-relative paths of all files under ``root``, ignore-filtered.

    The result is sorted for deterministic output. ``root`` must be an
    existing directory.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"workspace directory not found: {root}")
    matcher = IgnoreMatcher([*DEFAULT_IGNORES, *patterns])
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if matcher.matches(rel):
            continue
        files.append(rel)
    return files


class IgnoreMatcher:
    """Glob-based ignore rules over workspace-relative paths."""

    def __init__(self, patterns: Sequence[str]) -> None:
        expanded: list[str] = []
        for pattern in patterns:
            pattern = pattern.strip().rstrip("/")
            if not pattern:
                continue
            expanded.append(pattern)
            # A bare path (no wildcard) names a directory: ignore everything
            # under it, not just the directory itself.
            if not any(char in pattern for char in "*?["):
                expanded.append(f"{pattern}/**")
        self._patterns = tuple(expanded)

    def matches(self, rel: Path) -> bool:
        text = rel.as_posix()
        return any(fnmatch.fnmatch(text, pattern) for pattern in self._patterns)
