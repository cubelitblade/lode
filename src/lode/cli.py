"""lode CLI: survey / mine / prospect (aliases: status / index / search).

The mining metaphor carries the narrative (survey = detect, mine = embed,
prospect = search); the aliases keep the interface approachable for
practical use. MCP tools (index_status/reindex/search) are a thin layer on
the same functions — CLI first, MCP later (PLAN M1).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

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
    EmbedderUnavailableError,
    FileStatus,
    ModelStatus,
    SchemaVersionError,
    Store,
)
from lode.index.search import SearchHit, search
from lode.ingestion.pipeline import survey_workspace, sync
from lode.ingestion.split import RecursiveSegmentSplitter

# The index lives next to the workspace's own .lode/ directory (PLAN §3).
INDEX_DB_RELATIVE = Path(".lode") / "index.db"

app = typer.Typer(
    name="lode",
    help="lode: turn a workspace of documents into a searchable knowledge lode.",
    no_args_is_help=True,
)


def _model_gate(store: Store, *, rebuild_requested: bool = False) -> None:
    """Enforce the model-consistency contract for mine/prospect.

    A mismatch means the index was built with a different model: querying it
    is refused, updating it without a rebuild would silently keep stale
    vectors (unchanged files are skipped). UNKNOWN (embedding endpoint down)
    does not block — search must keep working (PLAN D7).
    """
    if store.model_status is ModelStatus.MISMATCH and not rebuild_requested:
        typer.echo(
            "The index was built with a different model "
            f"(indexed: {store.stored_model_id!r}). "
            "Run `lode mine --rebuild` to rebuild, or switch back to that model."
        )
        raise typer.Exit(code=1)


def _warn_stale(store: Store, hits: list[SearchHit]) -> None:
    """Warn when the index holds files that `lode survey` marked stale.

    Two distinct cases keep the message honest about where the risk sits:

    * no stale file shows up in this result set — stale data is lurking
      elsewhere; refresh keeps the library current;
    * a stale file does show up — those entries may not reflect the
      on-disk content, so verify before trusting them.
    """
    if not any(file.status is FileStatus.STALE for file in store.list_files()):
        return
    if any(hit.stale for hit in hits):
        typer.echo(
            "\nWarning: results include stale files; verify them before relying on them. "
            "Run `lode mine` to update the index."
        )
    else:
        typer.echo("\nWarning: the index holds stale files outside these results. Run `lode mine` to update the index.")


# Shared workspace argument shape; used by all three commands. The default
# value (".") is set with `=` at the call site, not inside `Argument`.
WorkspaceArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Workspace directory (default: current directory).",
    ),
]

# Shared config option: an explicit path skips auto-discovery (see config).
ConfigArg = Annotated[
    Path | None,
    typer.Option("--config", help="Path to a config file (skips auto-discovery)."),
]


@app.command("survey")
@app.command("status", hidden=True)
def survey(
    workspace: WorkspaceArg = Path("."),
    config: ConfigArg = None,
) -> None:
    """Detect workspace changes against the index; mark changed files stale.

    No embedding endpoint is needed — pure file comparison.
    """
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        typer.echo(f"Index needs a rebuild: {exc}")
        raise typer.Exit(code=1) from exc
    with store:
        result = survey_workspace(store, workspace, settings.ignore.sources)
        typer.echo(f"Surveyed {workspace}:")
        typer.echo(
            f"{result.unchanged} unchanged, {result.new} new, {result.changed} changed, "
            f"{result.missing} missing, {result.skipped} skipped."
        )
        if result.pending:
            typer.echo()
            typer.echo("Pending sync:")
            for path in result.new_files:
                typer.echo(f"  + {path}")
            for path in result.changed_files:
                typer.echo(f"  ~ {path}")
            for path in result.missing_files:
                typer.echo(f"  - {path}")
            typer.echo()
            typer.echo("Run `lode mine` to update index.")


@app.command("mine")
@app.command("index", hidden=True)
def mine(
    workspace: WorkspaceArg = Path("."),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Drop and rebuild the whole index first (required after a model change).",
    ),
    config: ConfigArg = None,
) -> None:
    """Embed changed/new files and update the index (requires an embedding endpoint).

    Use `--rebuild` after changing the embedding model or if the index
    schema is incompatible.
    """
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        typer.echo(f"Index needs a rebuild: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        _model_gate(store, rebuild_requested=rebuild)
        with store:
            if rebuild:
                store.rebuild()
            result = sync(
                store,
                workspace,
                embedder,
                RecursiveSegmentSplitter(
                    chunk_size=settings.chunking.size,
                    chunk_overlap=settings.chunking.overlap,
                ),
                settings.ignore.sources,
            )
            typer.echo(f"Mined {workspace}:")
            typer.echo(
                f"{result.added} added, {result.updated} updated, "
                f"{result.unchanged} unchanged, {result.removed} removed, "
                f"{result.skipped} skipped."
            )
            if result.added or result.updated or result.removed:
                typer.echo()
                for path in result.added_files:
                    typer.echo(f"  + {path}")
                for path in result.updated_files:
                    typer.echo(f"  ~ {path}")
                for path in result.removed_files:
                    typer.echo(f"  - {path}")
            if result.failed:
                typer.echo()
                typer.echo("Stumbled on:")
                for failure in result.failed:
                    typer.echo(f"  - {failure}")
                typer.echo("Re-run `lode mine` after fixing these to retry.")
    except EmbedderUnavailableError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc


@app.command("prospect")
@app.command("search", hidden=True)
def prospect(
    query: Annotated[str, typer.Argument(help="Query to search for.")],
    workspace: WorkspaceArg = Path("."),
    top_k: Annotated[int | None, typer.Option("--top-k", min=1, help="Number of results to return.")] = None,
    config: ConfigArg = None,
) -> None:
    """Search the index and show results with provenance (file + heading)."""
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        typer.echo(f"Index needs a rebuild: {exc}")
        raise typer.Exit(code=1) from exc

    _model_gate(store)
    with store:
        if top_k is None:
            top_k = settings.retrieval.top_k
        try:
            hits = search(
                store,
                embedder,
                query,
                semantic_weight=settings.retrieval.semantic_factor,
                lexical_weight=settings.retrieval.lexical_factor,
                top_k=top_k,
            )
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        if not hits:
            typer.echo("Dry hole: nothing matched.")
        else:
            for index, hit in enumerate(hits, start=1):
                stale_tag = " [stale]" if hit.stale else ""
                heading = f" > {hit.heading}" if hit.heading else ""
                page = f" (p.{hit.page})" if hit.page is not None else ""
                # Short content-address prefix keeps the line readable while
                # still identifying the chunk (full id lives in the DB).
                short_id = hit.chunk_id.removeprefix("blake3:")[:12]
                typer.echo(f"{index}. [{hit.score:.3f}] {hit.path}{heading}{page}{stale_tag} #{short_id}")
                snippet = hit.text.strip().replace("\n", " ")
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                typer.echo(f"   {snippet}")
        _warn_stale(store, hits)


@app.command("dig")
@app.command("get", hidden=True)
def dig(
    digest: Annotated[
        str,
        typer.Argument(help="Chunk digest: full `blake3:<hex>` or a short prefix (as shown by `prospect`)."),
    ],
    workspace: WorkspaceArg = Path("."),
    config: ConfigArg = None,
) -> None:
    """Fetch a chunk's full text by its digest (content address)."""
    db_path = workspace / INDEX_DB_RELATIVE
    if not db_path.exists():
        typer.echo(f"Dry hole: no index at {db_path}; run `lode mine` first.")
        raise typer.Exit(code=1)
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(db_path, embedder)
    except SchemaVersionError as exc:
        typer.echo(f"Index needs a rebuild: {exc}")
        raise typer.Exit(code=1) from exc
    with store:
        _dig(store, digest)


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


def _dig(store: Store, digest: str) -> None:
    token = _normalize_digest(digest)
    if not _DIGEST_PATTERN.fullmatch(token):
        typer.echo(f"Dry hole: not a valid digest: {digest!r}.")
        raise typer.Exit(code=1)
    matches = store.find_chunks_by_digest(token)
    if not matches:
        typer.echo(f"Dry hole: no chunk with digest {digest!r}.")
        raise typer.Exit(code=1)
    if len(matches) > 1:
        typer.echo(f"Digest {digest!r} is ambiguous ({len(matches)} chunks); use a longer prefix:")
        for chunk in matches:
            _echo_provenance(chunk)
        raise typer.Exit(code=1)
    _print_chunk(matches[0])


def _echo_provenance(chunk: ChunkWithPath) -> None:
    short_id = chunk.chunk_id.removeprefix("blake3:")[:12]
    heading = f" > {chunk.heading}" if chunk.heading else ""
    page = f" (p.{chunk.page})" if chunk.page is not None else ""
    typer.echo(f"  #{short_id} {chunk.path}{heading}{page}")


def _print_chunk(chunk: ChunkWithPath) -> None:
    short_id = chunk.chunk_id.removeprefix("blake3:")[:12]
    heading = f" > {chunk.heading}" if chunk.heading else ""
    page = f" (p.{chunk.page})" if chunk.page is not None else ""
    stale = " [stale]" if chunk.file_status is FileStatus.STALE else ""
    typer.echo(f"{chunk.path}{heading}{page}{stale} #{short_id}")
    typer.echo()
    typer.echo(chunk.text)


# -- `lode config`: read/write configuration without editing files -----------

# The scope of a `config` operation: a user-level global file or the project
# workspace file. Workspace is the default to keep lode's decentralised,
# project-first philosophy (no global state unless explicitly requested).
ConfigScope = Annotated[
    Literal["user", "workspace"],
    typer.Option("--scope", help="Config scope (default: workspace)."),
]

config_app = typer.Typer(
    name="config",
    help="Read and write configuration without editing files.",
    no_args_is_help=False,
)


def _target_config_path(scope: str) -> Path:
    return user_config_path() if scope == "user" else workspace_config_path()


def _format_config_value(key: str, value: Any) -> str:
    """Render a config value as `key = value` for `get` output."""
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif isinstance(value, (str, list)):
        rendered = json.dumps(value)
    else:
        rendered = str(value)
    return f"{key} = {rendered}"


def _cfg_show() -> None:
    """Print the merged effective configuration as TOML."""
    settings = load_settings()
    typer.echo(toml_dumps(effective_config(settings)))


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
    try:
        validate_key(key)
    except KeyError:
        typer.echo(f"Dry hole: unknown config key {key!r}.")
        raise typer.Exit(code=1) from None

    if scope is None:
        settings = load_settings()
        value = get_nested(effective_config(settings), key)
        typer.echo(_format_config_value(key, value))
        return
    if scope not in ("user", "workspace"):
        typer.echo("Stumbled: --scope must be 'user' or 'workspace'.")
        raise typer.Exit(code=1)

    path = _target_config_path(scope)
    data = read_toml(path)
    try:
        value = get_nested(data, key)
    except KeyError:
        typer.echo(f"Dry hole: {key!r} is not set in {path}.")
        raise typer.Exit(code=1) from None
    typer.echo(_format_config_value(key, value))


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.api.endpoint.")],
    value: Annotated[str, typer.Argument(help="Value to set; typed by the config field.")],
    scope: ConfigScope = "workspace",
) -> None:
    """Set a config value, inferring its type from the config field."""
    try:
        validate_key(key)
    except KeyError:
        typer.echo(f"Dry hole: unknown config key {key!r}.")
        raise typer.Exit(code=1) from None

    try:
        parsed = parse_value(key, value)
    except ValueError as exc:
        typer.echo(f"Stumbled: {exc}")
        raise typer.Exit(code=1) from exc

    path = _target_config_path(scope)
    data = read_toml(path)
    set_nested(data, key, parsed)
    write_toml(path, data)
    typer.echo(f"set {_format_config_value(key, parsed)} in {path}")


@config_app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help="Config key, e.g. embedding.api.endpoint.")],
    scope: ConfigScope = "workspace",
) -> None:
    """Unset a config value from the target scope."""
    try:
        validate_key(key)
    except KeyError:
        typer.echo(f"Dry hole: unknown config key {key!r}.")
        raise typer.Exit(code=1) from None

    path = _target_config_path(scope)
    data = read_toml(path)
    if not unset_nested(data, key):
        typer.echo(f"Dry hole: {key!r} is not set in {path}.")
        raise typer.Exit(code=1)
    write_toml(path, data)
    typer.echo(f"unset {key} in {path}")


@config_app.command("path")
def config_path(scope: ConfigScope = "workspace") -> None:
    """Show the target config file path (does not create the file)."""
    typer.echo(str(_target_config_path(scope)))


app.add_typer(config_app, name="config")


if __name__ == "__main__":
    app()
