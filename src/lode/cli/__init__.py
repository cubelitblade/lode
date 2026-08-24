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
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn

from lode.cli.commands._common import (
    block,
    dimension_mismatch_parts,
    render_options,
)
from lode.cli.render import Intent, RenderOptions
from lode.cli.render.config import (
    render_config_message,
    render_config_path,
    render_config_set,
    render_config_show,
    render_config_unset,
    render_config_value,
)
from lode.cli.render.dig import render_dig
from lode.cli.render.mine import render_mine
from lode.cli.render.output import echo_json, json_err, json_ok, preview, render_message
from lode.cli.render.prospect import render_prospect
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
from lode.embeddings.base import Embedder
from lode.index import (
    ChunkWithPath,
    DimensionMismatchError,
    EmbedderUnavailableError,
    ModelStatus,
    SchemaVersionError,
    Store,
)
from lode.index.search import ProspectResult, search
from lode.ingestion.pipeline import DetectResult, SyncSummary, detect_changes, sync
from lode.ingestion.split import RecursiveSegmentSplitter, SegmentSplitter

# The index lives next to the workspace's own .lode/ directory (PLAN §3).
INDEX_DB_RELATIVE = Path(".lode") / "index.db"


app = typer.Typer(
    name="lode",
    help="lode: turn a workspace of documents into a searchable knowledge lode.",
    no_args_is_help=True,
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
        block(message, code="model_mismatch", as_json=as_json, command=command)

    if from_scratch:
        return

    try:
        current_dimension = embedder.dimension
    except Exception:
        # Fault-tolerant like model detection: an unreachable endpoint must not
        # block search, which can still serve cached data (PLAN D7).
        return
    if current_dimension != store.dimension:
        error, hint = dimension_mismatch_parts(store.dimension, current_dimension)
        block(error, code="dimension_mismatch", as_json=as_json, command=command, hint=hint)


def _sync_with_progress(
    console: Console,
    store: Store,
    workspace: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    detect: DetectResult,
) -> SyncSummary:
    """Run ``sync`` while showing a live progress bar for observability.

    Rich's ``Progress`` needs a TTY, so the caller gates on
    ``console.is_terminal``. The bar shows the current file being processed, so
    it is clear the run is advancing rather than stuck on an embedding call.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("{task.description}"),
        console=console,
    )
    task_id: TaskID | None = None

    def report(processed: int, total: int, path: str) -> None:
        nonlocal task_id
        if task_id is None:
            if total == 0:
                # Nothing to mine; keep the live display empty.
                return
            task_id = progress.add_task("Mining", total=total)
        progress.update(task_id, completed=processed, description=path or "done")

    with progress:
        return sync(store, workspace, embedder, splitter, detect=detect, report=report)


# Shared workspace argument shape; only the help string varies per command.
# The default value (".") is set with `=` at the call site, not inside `Argument`.
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
    options = render_options(settings)
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
            splitter = RecursiveSegmentSplitter(
                chunk_size=settings.chunking.size,
                chunk_overlap=settings.chunking.overlap,
            )
            detect = detect_changes(store, workspace, settings.ignore.sources)
            console = Console()
            if detect.pending == 0:
                # Nothing to embed or prune; skip the progress bar entirely.
                result = SyncSummary()
                result.unchanged = detect.unchanged
                result.skipped = detect.skipped
            elif as_json:
                result = sync(store, workspace, embedder, splitter, detect=detect)
            elif console.is_terminal:
                result = _sync_with_progress(console, store, workspace, embedder, splitter, detect)
            else:
                result = sync(store, workspace, embedder, splitter, detect=detect)
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
                render_mine(workspace, result, console=console, options=options)
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
    """Search the index and show results with source information.

    Runs a silent detection first so the stale bits are fresh before search
    reads them — this command writes ``files.status`` (it is not read-only).
    """
    settings = load_settings(config)
    embedder = build_embedder(settings.embedding)
    options = render_options(settings)
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
        # Refresh the stale bits before searching so per-chunk annotation and
        # the library-wide dirty signal reflect the current workspace.
        detect = detect_changes(store, workspace, settings.ignore.sources)
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
            error, hint = dimension_mismatch_parts(exc.stored_dimension, exc.current_dimension)
            if as_json:
                echo_json(json_err("prospect", f"{error}\n{hint}", code="dimension_mismatch"))
            else:
                render_message(error, intent=Intent.ERROR)
                render_message(hint, intent=Intent.INFO)
            raise typer.Exit(code=1) from exc
        result = ProspectResult(
            workspace=workspace,
            query=query,
            top_k=top_k,
            hits=hits,
            has_stale=detect.dirty,
        )
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
                            "paths": [{"path": ref.path, "state": ref.status.value} for ref in hit.refs],
                            "heading": hit.heading,
                            "page": hit.page,
                            "digest": hit.digest,
                            "preview": preview(hit.text),
                        }
                        for index, hit in enumerate(hits, start=1)
                    ],
                )
            )
        else:
            render_prospect(result, options=options)


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
from lode.cli.commands import survey  # noqa: E402

survey.register(app)


if __name__ == "__main__":
    app()
