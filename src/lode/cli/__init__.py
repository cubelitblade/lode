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

from lode.cli.render.output import echo_json, json_err, json_ok, preview
from lode.cli.render.survey import render_survey
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
from lode.embeddings.base import Embedder
from lode.index import (
    ChunkWithPath,
    DimensionMismatchError,
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


def _block(message: str, *, code: str, as_json: bool, command: str) -> None:
    """Emit a blocking error (JSON or human) and exit non-zero."""
    if as_json:
        echo_json(json_err(command, message, code=code))
    else:
        typer.echo(message)
    raise typer.Exit(code=1)


def _dimension_mismatch_message(stored_dimension: int, current_dimension: int) -> str:
    """Friendly dimension-mismatch message with the two recovery options."""
    return (
        f"The current lode was driven at {stored_dimension} dimensions, but this model "
        f"yields {current_dimension}-wide nuggets — they don't sit on the same vein.\n"
        "Hint: Re-mine it with `lode mine --from-scratch`, or switch your embedding config "
        "back to the model/dimension that dug it."
    )


def _model_gate(
    store: Store,
    embedder: Embedder,
    *,
    from_scratch: bool = False,
    as_json: bool = False,
    command: str = "mine",
) -> None:
    """Enforce the model/dimension-consistency contract for mine/prospect.

    A mismatch means the index was built with a different model or a different
    vector dimension than the current embedder reports: querying it is refused
    and updating it without a from-scratch re-mine would silently keep stale or
    incompatible vectors. UNKNOWN (embedding endpoint down) does not block —
    search must keep working (PLAN D7).
    """
    if store.model_status is ModelStatus.MISMATCH and not from_scratch:
        message = (
            "The index was built with a different model "
            f"(indexed: {store.stored_model_id!r}). "
            "Run `lode mine --from-scratch` to re-mine it, or switch back to that model."
        )
        _block(message, code="model_mismatch", as_json=as_json, command=command)

    if from_scratch:
        return

    try:
        current_dimension = embedder.dimension
    except Exception:
        # Fault-tolerant like model detection: an unreachable endpoint must not
        # block search, which can still serve cached data (PLAN D7).
        return
    if current_dimension != store.dimension:
        message = _dimension_mismatch_message(store.dimension, current_dimension)
        _block(message, code="dimension_mismatch", as_json=as_json, command=command)


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


# Shared workspace argument shape; only the help string varies per command.
# The default value (".") is set with `=` at the call site, not inside `Argument`.
SurveyWorkspaceArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Workspace to inspect.",
    ),
]
MineWorkspaceArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Workspace to index.",
    ),
]
ProspectWorkspaceArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Workspace to search.",
    ),
]
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


@app.command("survey")
@app.command("status", hidden=True)
def survey(
    workspace: SurveyWorkspaceArg = Path("."),
    config: ConfigArg = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Detect workspace changes and report stale files.

    This command only compares files and does not require an embedding endpoint.
    """
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        message = f"Index needs a re-mine: {exc}"
        if as_json:
            echo_json(json_err("survey", message, code="schema_version"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1) from exc
    with store:
        result = survey_workspace(store, workspace, settings.ignore.sources)
        if as_json:
            echo_json(
                json_ok(
                    "survey",
                    workspace=str(workspace),
                    summary={
                        "unchanged": result.unchanged,
                        "new": result.new,
                        "changed": result.changed,
                        "missing": result.missing,
                        "skipped": result.skipped,
                        "pending": result.pending,
                    },
                    paths={
                        "new": result.new_files,
                        "changed": result.changed_files,
                        "missing": result.missing_files,
                        "unchanged": result.unchanged_files,
                    },
                )
            )
        else:
            render_survey(workspace, result)


@app.command("mine")
@app.command("index", hidden=True)
def mine(
    workspace: MineWorkspaceArg = Path("."),
    from_scratch: bool = typer.Option(
        False,
        "--from-scratch",
        help="Discard the existing index and create a new one from scratch.",
    ),
    config: ConfigArg = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Embeds and indexes new or changed files. An embedding endpoint must be configured.

    Use `--from-scratch` when changing the embedding model or when the index
    schema is incompatible.
    """
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        message = f"Index needs a re-mine: {exc}"
        if as_json:
            echo_json(json_err("mine", message, code="schema_version"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1) from exc

    try:
        _model_gate(store, embedder, from_scratch=from_scratch, as_json=as_json, command="mine")
        with store:
            if from_scratch:
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
            if as_json:
                echo_json(
                    json_ok(
                        "mine",
                        workspace=str(workspace),
                        from_scratch=from_scratch,
                        summary={
                            "added": result.added,
                            "updated": result.updated,
                            "unchanged": result.unchanged,
                            "removed": result.removed,
                            "skipped": result.skipped,
                        },
                        paths={
                            "added": result.added_files,
                            "updated": result.updated_files,
                            "removed": result.removed_files,
                        },
                        failed=[{"path": failure.path, "error": failure.error} for failure in result.failed],
                    )
                )
            else:
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
                        typer.echo(f"  - {failure.path}: {failure.error}")
                    typer.echo("Re-run `lode mine` after fixing these to retry.")
    except EmbedderUnavailableError as exc:
        if as_json:
            echo_json(json_err("mine", str(exc), code="embedder_unavailable"))
        else:
            typer.echo(str(exc))
        raise typer.Exit(code=1) from exc


@app.command("prospect")
@app.command("search", hidden=True)
def prospect(
    query: Annotated[str, typer.Argument(help="Query to search for.")],
    workspace: ProspectWorkspaceArg = Path("."),
    top_k: Annotated[int | None, typer.Option("--top-k", min=1, help="Maximum number of results to return.")] = None,
    config: ConfigArg = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Search the index and show results with source information."""
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        message = f"Index needs a re-mine: {exc}"
        if as_json:
            echo_json(json_err("prospect", message, code="schema_version"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1) from exc

    _model_gate(store, embedder, as_json=as_json, command="prospect")
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
            if as_json:
                echo_json(json_err("prospect", str(exc), code="invalid_query"))
            else:
                typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        except DimensionMismatchError as exc:
            # Fallback for a dimension mismatch the gate could not detect (e.g.
            # config dimension mirrors the stored value but the model actually
            # emits a different width): surface the friendly recovery message
            # instead of a raw sqlite traceback.
            message = _dimension_mismatch_message(exc.stored_dimension, exc.current_dimension)
            if as_json:
                echo_json(json_err("prospect", message, code="dimension_mismatch"))
            else:
                typer.echo(message)
            raise typer.Exit(code=1) from exc
        if as_json:
            echo_json(
                json_ok(
                    "prospect",
                    workspace=str(workspace),
                    query=query,
                    top_k=top_k,
                    hits=[
                        {
                            "rank": index,
                            "score": hit.score,
                            "path": hit.path,
                            "heading": hit.heading,
                            "page": hit.page,
                            "state": "stale" if hit.stale else "fresh",
                            "digest": hit.chunk_id,
                            "preview": preview(hit.text),
                        }
                        for index, hit in enumerate(hits, start=1)
                    ],
                )
            )
        else:
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
                    typer.echo(f"   {preview(hit.text)}")
            _warn_stale(store, hits)


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
        _dig(store, digest, as_json=as_json, radius=radius or 0)


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
        "digest": chunk.chunk_id,
        "path": chunk.path,
        "heading": chunk.heading,
        "page": chunk.page,
        "seq": chunk.seq,
        "state": "stale" if chunk.file_status is FileStatus.STALE else "fresh",
    }
    if include_text:
        data["text"] = chunk.text
    return data


def _dig(store: Store, digest: str, *, as_json: bool = False, radius: int = 0) -> None:
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
        _print_window(window_chunks, center_seq=target.seq, radius=radius)


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
    short_id = chunk.chunk_id.removeprefix("blake3:")[:12]
    heading = f" > {chunk.heading}" if chunk.heading else ""
    page = f" (p.{chunk.page})" if chunk.page is not None else ""
    typer.echo(f"  #{short_id} {chunk.path}{heading}{page}")


def _print_window(chunks: list[ChunkWithPath], *, center_seq: int | None, radius: int) -> None:
    """Print an ordered chunk window, marking the center chunk."""
    typer.echo(f"Window (center seq {center_seq}, radius {radius}):")
    for chunk in chunks:
        short_id = chunk.chunk_id.removeprefix("blake3:")[:12]
        heading = f" > {chunk.heading}" if chunk.heading else ""
        page = f" (p.{chunk.page})" if chunk.page is not None else ""
        stale = " [stale]" if chunk.file_status is FileStatus.STALE else ""
        marker = "[center] " if chunk.seq == center_seq else f"[seq {chunk.seq}] "
        typer.echo(f"{marker}{chunk.path}{heading}{page}{stale} #{short_id}")
        typer.echo()
        typer.echo(chunk.text)


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
