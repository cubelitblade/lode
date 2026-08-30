"""File discovery: walk a workspace and apply ignore rules.

Discovery is format-agnostic — it returns every file (as a
workspace-relative path) that survives the ignore rules. Whether a file can
actually be ingested is the extractor's decision, keeping the two concerns
separate.

Ignore rules use gitignore semantics (via ``pathspec``). ``.lodeignore`` is a
first-class citizen: it is always loaded when present at the workspace root,
without being listed in config. Config can name additional ignore-like files
(e.g. ``.gitignore``) under ``[app.ignore] sources``; all of them compose into a
single ruleset, the way git composes ``.gitignore`` files. The ignore files
themselves are excluded from the result — they are config, not content.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import pathspec

# Always ignored: the runtime data directory holding the index store.
DEFAULT_IGNORES = (".lode/**",)
# First-class ignore file, always loaded when present at the workspace root.
LODEIGNORE = ".lodeignore"


def discover(root: Path, ignore_files: Sequence[str] = ()) -> list[PurePosixPath]:
    """Workspace-relative paths of all files under ``root``, ignore-filtered.

    ``root`` must be an existing directory. ``ignore_files`` lists additional
    ignore-like files (read relative to ``root``), loaded after
    ``.lodeignore``. Ignore files themselves are never returned.

    Paths are returned as the platform-independent domain type
    (:class:`~pathlib.PurePosixPath`, see ``lode.relpath``).
    """
    if not root.is_dir():
        raise FileNotFoundError(f"workspace directory not found: {root}")

    spec = _build_ignore_spec(root, ignore_files)
    excluded = _ignore_file_paths(ignore_files)

    files: list[PurePosixPath] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_text = path.relative_to(root).as_posix()
        if rel_text in excluded:
            continue
        if spec.match_file(rel_text):
            continue
        files.append(PurePosixPath(rel_text))
    return files


def _build_ignore_spec(root: Path, ignore_files: Sequence[str]) -> pathspec.GitIgnoreSpec:
    """Compose a gitignore-style spec from defaults + .lodeignore + config files."""
    lines: list[str] = list(DEFAULT_IGNORES)

    lodeignore = root / LODEIGNORE
    if lodeignore.is_file():
        lines.extend(_read_ignore_lines(lodeignore))

    for name in ignore_files:
        path = root / name
        if path.is_file():
            lines.extend(_read_ignore_lines(path))

    return pathspec.GitIgnoreSpec.from_lines(lines)


def _read_ignore_lines(path: Path) -> list[str]:
    """Return the lines of an ignore file; pathspec handles comments and blanks."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def _ignore_file_paths(ignore_files: Sequence[str]) -> set[str]:
    """Workspace-relative paths of the ignore files, so they never get indexed."""
    return {Path(name).as_posix() for name in (LODEIGNORE, *ignore_files)}
