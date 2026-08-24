"""lode CLI: survey / mine / prospect (aliases: status / index / search).

The mining metaphor carries the narrative (survey = detect, mine = embed,
prospect = search); the aliases keep the interface approachable for
practical use. MCP tools (index_status/reindex/search) are a thin layer on
the same functions — CLI first, MCP later (PLAN M1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from lode.cli.commands._common import render_options
from lode.cli.render.config import (
    render_config_message,
    render_config_path,
    render_config_set,
    render_config_show,
    render_config_unset,
    render_config_value,
)
from lode.cli.render.dig import render_dig as render_dig  # re-exported for lode.cli.render_dig
from lode.cli.render.mine import render_mine as render_mine  # re-exported for lode.cli.render_mine
from lode.cli.render.prospect import render_prospect as render_prospect  # re-exported for lode.cli.render_prospect
from lode.cli.render.survey import render_survey as render_survey  # re-exported for lode.cli.render_survey
from lode.config import (
    build_embedder as build_embedder,  # re-exported for lode.cli.build_embedder
)
from lode.config import (
    effective_config,
    get_nested,
    load_settings,
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

app = typer.Typer(
    name="lode",
    help="lode: turn a workspace of documents into a searchable knowledge lode.",
    no_args_is_help=True,
)


# -- `lode config`: read/write configuration without editing files -----------

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


def _cfg_show() -> None:
    """Print the merged effective configuration as TOML."""
    settings = load_settings()
    render_config_show(toml_dumps(effective_config(settings)), options=render_options(settings))


@config_app.callback(invoke_without_command=True)
def _config_callback(ctx: typer.Context) -> None:  # pyright: ignore[reportUnusedFunction]  # registered as the sub-app callback; not called directly
    # Bare `lode config` (no subcommand) is an alias for `show`.
    if ctx.invoked_subcommand is None:
        _cfg_show()


@config_app.command("show")
def config_show() -> None:
    """Show the merged effective configuration (TOML)."""
    _cfg_show()


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.api.endpoint.")],
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
    settings = load_settings()
    options = render_options(settings)
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
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.api.endpoint.")],
    value: Annotated[str, typer.Argument(help="Value to set; typed by the config field.")],
    scope: ConfigScope = "workspace",
) -> None:
    """Set a config value, inferring its type from the config field."""
    settings = load_settings()
    options = render_options(settings)
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
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.api.endpoint.")],
    scope: ConfigScope = "workspace",
) -> None:
    """Unset a config value from the target scope."""
    settings = load_settings()
    options = render_options(settings)
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
def config_path(scope: ConfigScope = "workspace") -> None:
    """Show the target config file path (does not create the file)."""
    render_config_path(_target_config_path(scope), options=render_options(load_settings()))


app.add_typer(config_app, name="config")

# Register the command modules on the shared app. Imported at the bottom so
# the module-level names above (build_embedder, render_*, ...) are bound
# before the command modules resolve them through `lode.cli`.
from lode.cli.commands import dig, mine, prospect, survey  # noqa: E402

survey.register(app)
mine.register(app)
prospect.register(app)
dig.register(app)


if __name__ == "__main__":
    app()
