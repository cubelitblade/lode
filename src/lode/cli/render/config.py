"""Human-readable `lode config` rendering.

Config output is *not* a structured report: ``show`` dumps raw TOML and the
other subcommands print single-line values/confirmations. These are emitted
verbatim (no wrapping, no token re-colouring) so the TOML/formatting is
preserved, while status/error lines still take an ``Intent`` colour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from lode.cli.render.core import Intent, RenderOptions


def _format_config_value(key: str, value: Any) -> str:
    """Render a config value as `key = value` for `get`/`set` output."""
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif isinstance(value, (str, list)):
        rendered = json.dumps(value)
    else:
        rendered = str(value)
    return f"{key} = {rendered}"


def _print_verbatim(console: Console, content: str) -> None:
    """Print content untouched: no word-wrap, no token re-colouring."""
    console.print(content, markup=False, highlight=False, soft_wrap=True)


def _print_intent(console: Console, content: str, *, options: RenderOptions, intent: Intent) -> None:
    """Print a single line verbatim, coloured by ``intent``."""
    console.print(
        content,
        style=options.intent_colors.get(intent, ""),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def render_config_show(content: str, *, options: RenderOptions | None = None, console: Console | None = None) -> None:
    """Render the merged configuration as verbatim TOML (no wrapping)."""
    console = console or Console(no_color=(options or RenderOptions()).no_color)
    _print_verbatim(console, content)


def render_config_value(
    key: str,
    value: Any,
    *,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a config value as ``key = value`` (verbatim)."""
    console = console or Console(no_color=(options or RenderOptions()).no_color)
    _print_verbatim(console, _format_config_value(key, value))


def render_config_set(
    key: str,
    value: Any,
    path: Path,
    *,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a ``config set`` confirmation."""
    console = console or Console(no_color=(options or RenderOptions()).no_color)
    _print_intent(
        console,
        f"set {_format_config_value(key, value)} in {path}",
        options=options or RenderOptions(),
        intent=Intent.INFO,
    )


def render_config_unset(
    key: str,
    path: Path,
    *,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a ``config unset`` confirmation."""
    console = console or Console(no_color=(options or RenderOptions()).no_color)
    _print_intent(
        console,
        f"unset {key} in {path}",
        options=options or RenderOptions(),
        intent=Intent.INFO,
    )


def render_config_path(path: Path, *, options: RenderOptions | None = None, console: Console | None = None) -> None:
    """Render the target config file path (verbatim)."""
    console = console or Console(no_color=(options or RenderOptions()).no_color)
    _print_verbatim(console, str(path))


def render_config_message(
    message: str,
    *,
    intent: Intent = Intent.ERROR,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a config error/status line with an ``Intent`` colour."""
    console = console or Console(no_color=(options or RenderOptions()).no_color)
    _print_intent(console, message, options=options or RenderOptions(), intent=intent)
