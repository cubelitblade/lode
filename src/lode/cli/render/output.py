"""Shared output primitives for lode CLI.

The machine-readable ``--json`` envelope helpers, the preview truncation used
by ``prospect``/``dig``, and the ``render_message`` primitive that emits a
single human-readable line with an ``Intent`` colour. Command-specific reports
live in per-command modules (``survey.py``, ``mine.py``, ...).
"""

from __future__ import annotations

import json
import re
from typing import Any

import typer
from rich.console import Console

from lode.cli.render.core import Intent, RenderOptions

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


def render_message(
    message: str,
    *,
    intent: Intent,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a single human-readable line with the given ``Intent`` colour.

    For non-command human output (errors, warnings, hints) that does not
    warrant a full per-command renderer. Colour comes from
    ``options.intent_colors`` and is stripped in non-TTY runs; ``console`` may
    be injected (e.g. a recording console) to capture output in tests.

    ``markup`` and ``highlight`` are disabled so the message is emitted verbatim
    and rich does not re-colour tokens such as numbers (which would otherwise
    override the ``intent`` colour).
    """
    if options is None:
        options = RenderOptions()
    console = console or Console(no_color=options.no_color)
    style = options.intent_colors.get(intent, "")
    console.print(message, style=style, markup=False, highlight=False)
