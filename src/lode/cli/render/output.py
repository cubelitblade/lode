"""Shared output primitives for lode CLI.

The machine-readable ``--json`` envelope helpers and the preview truncation
used by ``prospect``/``dig``. Human-readable rendering lives in per-command
modules (``survey.py``, and later ``mine.py``/``prospect.py``/``dig.py``).
"""

from __future__ import annotations

import json
import re
from typing import Any

import typer

# Machine-readable output (--json) envelope. This is the bridge to a future
# MCP layer: every `--json`-capable command emits the same top-level shape
# (schema_version, command, success, plus data or error) so consumers can tell
# success/failure apart and read data uniformly. `schema_version` lets the
# shape evolve without breaking existing consumers.
JSON_SCHEMA_VERSION = 1


def json_ok(command: str, **data: Any) -> dict[str, Any]:
    """Build a successful --json envelope (success=True + command data)."""
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": command,
        "success": True,
        **data,
    }


def json_err(command: str, message: str, *, code: str = "error", **error_extra: Any) -> dict[str, Any]:
    """Build a failed --json envelope (success=False + structured error).

    ``error_extra`` lets callers add fields to the ``error`` object (e.g.
    ``candidates`` for an ambiguous ``dig``), keeping the envelope uniform.
    """
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": command,
        "success": False,
        "error": {"code": code, "message": message, **error_extra},
    }


def echo_json(payload: dict[str, Any]) -> None:
    """Print a --json payload as an indented, non-ASCII-safe JSON document."""
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


# Max length of the `prospect` preview snippet. Human output and --json share
# it so the two never drift; may become a config option later.
PREVIEW_MAX_CHARS = 160


def preview(text: str) -> str:
    """Flatten text to a single line, truncated to PREVIEW_MAX_CHARS.

    Line breaks (\r, \n) and other whitespace runs collapse to a single space.
    """
    snippet = re.sub(r"\s+", " ", text).strip()
    if len(snippet) > PREVIEW_MAX_CHARS:
        snippet = snippet[: PREVIEW_MAX_CHARS - 3] + "..."
    return snippet
