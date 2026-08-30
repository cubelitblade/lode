"""Workspace-relative path conventions.

Workspace-relative paths are platform-independent values (``PurePosixPath``):
they live in the domain model, the index database, and JSON payloads, and must
byte-match across machines (e.g. WSL reading a Windows checkout through
``/mnt/c``). Only the human-facing render layer converts them to OS-native
``Path`` objects via :func:`to_native`.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def to_rel(path: Path | str | PurePosixPath) -> PurePosixPath:
    """Normalize a workspace-relative path to the platform-independent domain type.

    Accepts an OS-native ``Path`` (as produced by filesystem walks), posix
    text (as stored in the database), or an already-normalized value.
    """
    if isinstance(path, Path):
        return PurePosixPath(path.as_posix())
    return PurePosixPath(path)


def to_native(rel: PurePosixPath) -> Path:
    """Convert to an OS-native path for disk I/O or human display.

    Rebuilds from parts instead of passing the pure path to the ``Path``
    constructor, so no behaviour depends on mixing path flavours.
    """
    return Path(*rel.parts)
