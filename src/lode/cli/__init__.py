"""lode CLI: survey / mine / prospect (aliases: status / index / search).

The mining metaphor carries the narrative (survey = detect, mine = embed,
prospect = search); the aliases keep the interface approachable for
practical use. MCP tools (index_status/reindex/search) are a thin layer on
the same functions — CLI first, MCP later (PLAN M1).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from lode.cli.commands._common import (
    render_options,
)
from lode.cli.render import RenderOptions
from lode.cli.render.config import (
    render_config_message,
    render_config_path,
    render_config_set,
    render_config_show,
    render_config_unset,
    render_config_value,
)
from lode.cli.render.dig import render_dig
from lode.cli.render.mine import render_mine as render_mine  # re-exported for lode.cli.render_mine
from lode.cli.render.output import echo_json, json_err, json_ok
from lode.cli.render.prospect import render_prospect as render_prospect  # re-exported for lode.cli.render_prospect
from lode.cli.render.survey import render_survey as render_survey  # re-exported for lode.cli.render_survey
from lode.config import (
    build_embedder,
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
from lode.index import (
    ChunkWithPath,
    SchemaVersionError,
    Store,
)

# The index lives next to the workspace's own .lode/ directory (PLAN §3).
INDEX_DB_RELATIVE = Path(".lode") / "index.db"


app = typer.Typer(
    name="lode",
    help="lode: turn a workspace of documents into a searchable knowledge lode.",
    no_args_is_help=True,
)


# Shared workspace argument shape; only the help string varies per command.
# The default value (".") is set with `=` at the call site, not inside `Argument`.
DigWorkspaceArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Workspace containing the index.",
    ),
]

# Shared config option: an explicit path that skips auto-discovery (see config).
ConfigArg = Annotated[
    Path | None,
    typer.Option("--config", help="Path to a configuration file."),
]


@app.command("dig")
@app.command("get", hidden=True)
def dig(
    digest: Annotated[
        str,
        typer.Argument(help="Chunk digest or prefix."),
    ],
    workspace: DigWorkspaceArg = Path("."),
    radius: Annotated[
        int | None,
        typer.Option(
            "--radius",
            min=0,
            help="Number of adjacent chunks to include around the target chunk.",
        ),
    ] = None,
    config: ConfigArg = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Fetch a chunk's full text by its digest (content address)."""
    db_path = workspace / INDEX_DB_RELATIVE
    if not db_path.exists():
        message = f"Dry hole: no index at {db_path}; run `lode mine` first."
        if as_json:
            echo_json(json_err("dig", message, code="no_index"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1)
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    options = render_options(settings)
    try:
        store = Store(db_path, embedder)
    except SchemaVersionError as exc:
        message = f"Index needs a re-mine: {exc}"
        if as_json:
            echo_json(json_err("dig", message, code="schema_version"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1) from exc
    with store:
        _dig(store, digest, as_json=as_json, radius=radius or 0, options=options)


# Hex hexdigest bodies (BLAKE3 produces lowercase hex); accept uppercase too.
_DIGEST_PATTERN = re.compile(r"[0-9a-f]+", re.IGNORECASE)


def _normalize_digest(digest: str) -> str:
    """Strip cosmetics that carry no addressing value.

    Accepts the full ``blake3:<hex>``, the bare hex, or the short prefix
    ``prospect`` prints (including a leading ``#``). Any uppercase hex is
    folded to lowercase so it matches stored content addresses.
    """
    token = digest.strip()
    if token.startswith("#"):
        token = token[1:]
    return token.removeprefix("blake3:").lower()


def _chunk_to_json(chunk: ChunkWithPath, *, include_text: bool = True) -> dict[str, Any]:
    """Build a `dig` --json payload for one chunk (omit text for candidates)."""
    data: dict[str, Any] = {
        "digest": chunk.digest,
        "paths": [{"path": ref.path, "state": ref.status.value} for ref in chunk.refs],
        "heading": chunk.heading,
        "page": chunk.page,
        "seq": chunk.seq,
    }
    if include_text:
        data["text"] = chunk.text
    return data


def _dig(
    store: Store,
    digest: str,
    *,
    as_json: bool = False,
    radius: int = 0,
    options: RenderOptions | None = None,
) -> None:
    token = _normalize_digest(digest)
    if not _DIGEST_PATTERN.fullmatch(token):
        message = f"Dry hole: not a valid digest: {digest!r}."
        if as_json:
            echo_json(json_err("dig", message, code="invalid_digest"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1)
    matches = store.find_chunks_by_digest(token)
    if not matches:
        message = f"Dry hole: no chunk with digest {digest!r}."
        if as_json:
            echo_json(json_err("dig", message, code="not_found"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1)
    if len(matches) > 1:
        message = f"Digest {digest!r} is ambiguous ({len(matches)} chunks); use a longer prefix:"
        if as_json:
            echo_json(
                json_err(
                    "dig",
                    message,
                    code="ambiguous",
                    candidates=[_chunk_to_json(chunk, include_text=False) for chunk in matches],
                )
            )
        else:
            typer.echo(message)
            for chunk in matches:
                _echo_provenance(chunk)
        raise typer.Exit(code=1)

    target = matches[0]
    window_chunks = _window(store, token, target, radius)
    if as_json:
        echo_json(
            json_ok(
                "dig",
                window={
                    "center_seq": target.seq,
                    "radius": radius,
                    "chunks": [_chunk_to_json(chunk) for chunk in window_chunks],
                },
            )
        )
    else:
        render_dig(
            window_chunks,
            digest=target.digest.removeprefix("blake3:")[:12],
            center_seq=target.seq,
            radius=radius,
            options=options,
        )


def _window(store: Store, token: str, target: ChunkWithPath, radius: int) -> list[ChunkWithPath]:
    """Build the ordered chunk window: target plus same-section neighbors.

    Neighbors are constrained to the same file/heading and within ``radius``
    on each side; the window is returned sorted by ``seq`` so the caller can
    splice it directly. ``radius`` of 0 yields just the target chunk.
    """
    chunks = [target, *store.get_chunk_neighbors(token, radius)]
    chunks.sort(key=lambda chunk: chunk.seq if chunk.seq is not None else -1)
    return chunks


def _echo_provenance(chunk: ChunkWithPath) -> None:
    short_id = chunk.digest.removeprefix("blake3:")[:12]
    more = f" (+{len(chunk.refs) - 1} more)" if len(chunk.refs) > 1 else ""
    heading = f" > {chunk.heading}" if chunk.heading else ""
    page = f" (p.{chunk.page})" if chunk.page is not None else ""
    typer.echo(f"  #{short_id} {chunk.primary.path}{more}{heading}{page}")


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
from lode.cli.commands import mine, prospect, survey  # noqa: E402

survey.register(app)
mine.register(app)
prospect.register(app)


if __name__ == "__main__":
    app()
