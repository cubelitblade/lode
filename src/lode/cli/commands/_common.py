"""Shared plumbing for the lode CLI command modules.

The command layer's composition seam: store opening (uniform no-index
short-circuit and error exit), the symmetric error exit, the model/dimension
gate, progress rendering, and the shared argument shapes. Command modules
import from here instead of re-implementing these.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Annotated, Literal, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn

from lode.cli.render import Intent, RenderOptions, render_options_from_preset
from lode.cli.render.output import echo_json, json_err, render_message
from lode.config import DEFAULT_TOKENIZER, Settings, load_settings
from lode.embeddings.base import Embedder
from lode.index import (
    EmbedderUnavailableError,
    Store,
    StoreError,
    check_index_compatibility,
    read_index_meta,
)
from lode.ingestion.pipeline import DetectResult, SyncSummary, sync
from lode.ingestion.split import SegmentSplitter
from lode.messages import error_text, require_error_text

# The index lives next to the workspace's own .lode/ directory.
INDEX_DB_RELATIVE = Path(".lode") / "index.db"

# Hex hexdigest bodies (BLAKE3 produces lowercase hex); accept uppercase too.
# Shared by `dig` and `assay` for resolving a digest or short prefix.
DIGEST_PATTERN = re.compile(r"[0-9a-f]+", re.IGNORECASE)


def normalize_digest(digest: str) -> str:
    """Strip cosmetics that carry no addressing value.

    Accepts the full ``blake3:<hex>``, the bare hex, or the short prefix
    ``prospect`` prints (including a leading ``#``). Any uppercase hex is
    folded to lowercase so it matches stored content addresses.
    """
    token = digest.strip()
    if token.startswith("#"):
        token = token[1:]
    return token.removeprefix("blake3:").lower()


def block(
    message: str,
    *,
    code: str,
    as_json: bool,
    command: str,
    intent: Intent = Intent.ERROR,
    hint: str | None = None,
    options: RenderOptions | None = None,
) -> NoReturn:
    """Emit a blocking error (JSON or human) and exit non-zero.

    Human output renders ``message`` with ``intent`` and an optional ``hint``
    with ``INFO``; JSON output joins the two into one envelope. ``options``
    carries the resolved output styling so error lines honour palette and
    no-color like the rest of the command's output.
    """
    if as_json:
        full = f"{message}\n{hint}" if hint else message
        echo_json(json_err(command, full, code=code))
    else:
        render_message(message, intent=intent, options=options)
        if hint:
            render_message(hint, intent=Intent.INFO, options=options)
    raise typer.Exit(code=1)


def fail_with(
    code: str,
    *,
    command: str,
    as_json: bool,
    options: RenderOptions | None = None,
    **fields: object,
) -> NoReturn:
    """Emit a fixed-shape failure from the message table and exit non-zero.

    The wording comes from the ``lode.messages`` entry for ``code``; the same
    code labels the JSON envelope.
    """
    text = require_error_text(code, **fields)
    block(text.error, code=code, as_json=as_json, command=command, hint=text.hint or None, options=options)


def load_settings_or_fail(
    config: Path | str | None,
    *,
    command: str,
    as_json: bool,
) -> Settings:
    """Load settings, mapping config failures to a friendly blocking exit.

    A malformed TOML file, an explicit path that does not exist, or a value
    that fails validation would otherwise surface as a raw traceback; here
    they become one readable line instead.
    """
    try:
        return load_settings(config)
    except (ValidationError, TOMLDecodeError, FileNotFoundError) as exc:
        block(
            require_error_text("config_invalid", detail=str(exc)).error,
            code="config_invalid",
            as_json=as_json,
            command=command,
        )


def store_failure(exc: Exception, *, command: str, as_json: bool) -> NoReturn:
    """Emit a store/embedder failure through the shared blocking exit.

    User-facing wording comes from the ``lode.messages`` table, looked up by
    the exception's stable ``code`` and filled from its ``template_fields()``.
    Codes without a template fall back to the exception's diagnostic message.
    """
    code = getattr(exc, "code", "store_error")
    fields = exc.template_fields() if isinstance(exc, StoreError) else {}
    text = error_text(code, **fields)
    if text is None:
        block(str(exc), code=code, as_json=as_json, command=command)
    block(text.error, code=code, as_json=as_json, command=command, hint=text.hint or None)


def open_store(
    workspace: Path,
    embedder: Embedder | None,
    *,
    command: str,
    as_json: bool,
    tokenizer: str = DEFAULT_TOKENIZER,
) -> Store | None:
    """Open the workspace index, or return ``None`` if it does not exist yet.

    Only opens an existing database — creating one is ``mine``'s job. Before
    a ``Store`` is constructed, the index header is classified against the
    current configuration; any mismatch (schema version, model, dimension,
    tokenizer) exits through the shared failure path with its friendly
    template. Callers that need to destroy a mismatched index (e.g.
    ``mine --from-scratch``) must do so *before* calling this function.
    """
    db_path = workspace / INDEX_DB_RELATIVE
    if not db_path.exists():
        return None
    meta = read_index_meta(db_path)
    issues = check_index_compatibility(meta, embedder=embedder, tokenizer=tokenizer)
    if issues:
        issue = issues[0]
        text = require_error_text(issue.code, **issue.fields)
        block(text.error, code=issue.code, as_json=as_json, command=command, hint=text.hint or None)
    try:
        return Store(db_path, embedder, tokenizer=tokenizer, meta=meta)
    except (StoreError, EmbedderUnavailableError) as exc:
        store_failure(exc, command=command, as_json=as_json)


def require_store(
    workspace: Path,
    embedder: Embedder | None,
    *,
    command: str,
    as_json: bool,
    tokenizer: str,
    options: RenderOptions,
) -> Store:
    """Open the workspace index, exiting with ``no_index`` if it does not exist.

    Wraps ``open_store`` + the ``no_index`` short-circuit shared by the
    read-oriented commands (assay/dig/prospect). Returns a non-None ``Store``;
    a missing index exits through the shared failure path. ``tokenizer`` and
    ``options`` are required: every caller resolves them from settings, and a
    default would silently mask a caller that forgot to pass them.
    """
    store = open_store(workspace, embedder, command=command, as_json=as_json, tokenizer=tokenizer)
    if store is None:
        fail_with(
            "no_index",
            command=command,
            as_json=as_json,
            options=options,
            index_path=str(workspace / INDEX_DB_RELATIVE),
        )
    return store


def resolve_render_options(
    *,
    configured_palette: str,
    configured_no_color: bool | None,
    palette: str | None = None,
    no_color: bool = False,
) -> RenderOptions:
    """Resolve palette + no_color (flag overrides config) to render options.

    ``palette`` (``vivid``/``accessible``) selects ``intent_colors``; ``no_color``
    is an independent on/off switch applied at the ``Console`` layer. The flag
    (``--no-color``) wins over config; when neither is set it stays ``None`` so
    Rich's own ``NO_COLOR`` detection applies.
    """
    name = palette if palette is not None else configured_palette
    preset = render_options_from_preset(name)
    resolved_no_color: bool | None = True if no_color else configured_no_color
    return replace(preset, no_color=resolved_no_color)


def render_options(
    settings: Settings,
    palette: str | None = None,
    no_color: bool = False,
) -> RenderOptions:
    """Resolve the configured output palette (plus flag overrides) to options.

    The command layer owns this resolution and hands the render layer a
    ready-made ``RenderOptions``; the render layer never reads config directly.
    """
    return resolve_render_options(
        configured_palette=settings.output.palette,
        configured_no_color=settings.output.no_color,
        palette=palette,
        no_color=no_color,
    )


def sync_with_progress(
    console: Console,
    store: Store,
    workspace: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    detect: DetectResult,
) -> SyncSummary:
    """Run ``sync`` while showing a live progress bar (TTY only).

    The bar shows the current file being processed, so it is clear the run is
    advancing rather than stuck on an embedding call.
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


def workspace_from(ctx: typer.Context) -> Path:
    """Resolve the workspace from the application-level callback.

    ``lode --workspace <path> <command>`` stores the resolved ``Path`` (default
    ``.``) on the click context; commands read it here instead of declaring
    their own argument. The callback already validated the path (exists,
    directory), so this is a plain read.
    """
    value = ctx.find_object(Path)
    return value if value is not None else Path(".")


# Shared config option: an explicit path that skips auto-discovery (see config).
ConfigArg = Annotated[
    Path | None,
    typer.Option("--config", help="Path to a configuration file."),
]

# Shared output options: a per-run palette override and a no-colour switch.
# Both default to "unset" so the configured value wins unless the user passes
# them; `--no-color` takes precedence over `--palette`.
PaletteArg = Annotated[
    Literal["vivid", "accessible"] | None,
    typer.Option("--palette", help="Colour palette for this run."),
]
NoColorArg = Annotated[
    bool,
    typer.Option("--no-color", help="Disable colour for this run."),
]
