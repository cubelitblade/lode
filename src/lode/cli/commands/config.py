"""The ``config`` sub-application: read/write configuration without editing files.

A separate typer sub-app registered on the main app. ``show`` prints the
merged effective config as TOML; ``get``/``set``/``unset``/``path`` operate
on a user or workspace scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from lode.cli.commands._common import NoColorArg, PaletteArg, load_settings_or_fail, render_options
from lode.cli.render import RenderOptions
from lode.cli.render.config import (
    render_config_message,
    render_config_path,
    render_config_set,
    render_config_show,
    render_config_unset,
    render_config_value,
)
from lode.config import (
    effective_config,
    get_nested,
    parse_value,
    read_toml,
    set_nested,
    toml_dumps,
    unset_nested,
    user_config_path,
    validate_key,
    workspace_config_path,
    write_toml,
)

# The scope of a `config` operation: a user-level global file or the project
# workspace file. Workspace is the default to keep lode's decentralised,
# project-first philosophy (no global state unless explicitly requested).
ConfigScope = Annotated[
    Literal["user", "workspace"],
    typer.Option("--scope", help="Config scope."),
]

config_app = typer.Typer(
    name="config",
    help="Inspect and modify configuration.",
    no_args_is_help=False,
)


def _target_config_path(scope: str) -> Path:
    return user_config_path() if scope == "user" else workspace_config_path()


def _cfg_show(options: RenderOptions) -> None:
    """Print the merged effective configuration as TOML."""
    settings = load_settings_or_fail(None, command="config", as_json=False)
    render_config_show(toml_dumps(effective_config(settings)), options=options)


@config_app.callback(invoke_without_command=True)
def _config_callback(  # pyright: ignore[reportUnusedFunction]  # registered as the sub-app callback; not called directly
    ctx: typer.Context,
    palette: PaletteArg = None,
    no_color: NoColorArg = False,
) -> None:
    # Resolve output options once for the whole sub-app; subcommands read them
    # from ctx.obj. Bare `lode config` (no subcommand) is an alias for `show`.
    options = render_options(load_settings_or_fail(None, command="config", as_json=False), palette, no_color)
    ctx.obj = options
    if ctx.invoked_subcommand is None:
        _cfg_show(options)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show the merged effective configuration (TOML)."""
    _cfg_show(cast(RenderOptions, ctx.obj))


@config_app.command("get")
def config_get(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.openai_compatible.endpoint.")],
    scope: Annotated[
        str | None,
        typer.Option("--scope", help="Read only this scope; omit to read the merged effective value."),
    ] = None,
) -> None:
    """Read a config value.

    Without `--scope` this returns the merged effective value (the same one
    used at runtime). With `--scope user|workspace` it returns that layer's
    explicit value, failing if it is not set there.
    """
    options = cast(RenderOptions, ctx.obj)
    settings = load_settings_or_fail(None, command="config", as_json=False)
    try:
        validate_key(key)
    except KeyError:
        render_config_message(f"Dry hole: unknown config key {key!r}.", options=options)
        raise typer.Exit(code=1) from None

    if scope is None:
        value = get_nested(effective_config(settings), key)
        render_config_value(key, value, options=options)
        return
    if scope not in ("user", "workspace"):
        render_config_message("Stumbled: --scope must be 'user' or 'workspace'.", options=options)
        raise typer.Exit(code=1)

    path = _target_config_path(scope)
    data = read_toml(path)
    try:
        value = get_nested(data, key)
    except KeyError:
        render_config_message(f"Dry hole: {key!r} is not set in {path}.", options=options)
        raise typer.Exit(code=1) from None
    render_config_value(key, value, options=options)


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.openai_compatible.endpoint.")],
    value: Annotated[str, typer.Argument(help="Value to set; typed by the config field.")],
    scope: ConfigScope = "workspace",
) -> None:
    """Set a config value, inferring its type from the config field."""
    options = cast(RenderOptions, ctx.obj)
    try:
        validate_key(key)
    except KeyError:
        render_config_message(f"Dry hole: unknown config key {key!r}.", options=options)
        raise typer.Exit(code=1) from None

    try:
        parsed = parse_value(key, value)
    except ValueError as exc:
        render_config_message(f"Stumbled: {exc}", options=options)
        raise typer.Exit(code=1) from exc

    path = _target_config_path(scope)
    data = read_toml(path)
    set_nested(data, key, parsed)
    write_toml(path, data)
    render_config_set(key, parsed, path, options=options)


@config_app.command("unset")
def config_unset(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.openai_compatible.endpoint.")],
    scope: ConfigScope = "workspace",
) -> None:
    """Unset a config value from the target scope."""
    options = cast(RenderOptions, ctx.obj)
    try:
        validate_key(key)
    except KeyError:
        render_config_message(f"Dry hole: unknown config key {key!r}.", options=options)
        raise typer.Exit(code=1) from None

    path = _target_config_path(scope)
    data = read_toml(path)
    if not unset_nested(data, key):
        render_config_message(f"Dry hole: {key!r} is not set in {path}.", options=options)
        raise typer.Exit(code=1)
    write_toml(path, data)
    render_config_unset(key, path, options=options)


@config_app.command("path")
def config_path(ctx: typer.Context, scope: ConfigScope = "workspace") -> None:
    """Show the target config file path (does not create the file)."""
    render_config_path(_target_config_path(scope), options=cast(RenderOptions, ctx.obj))
