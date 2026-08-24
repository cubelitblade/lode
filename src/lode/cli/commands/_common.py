"""Shared plumbing for the lode CLI command modules.

This is the command layer's composition seam: store opening (with a uniform
no-index short-circuit and error exit), the symmetric error exit, the
model/dimension gate, progress rendering, and the shared argument shapes.
Command modules import from here instead of re-implementing these.

The seam is the E1/E2 fix: ``SchemaVersionError`` handling and the error
exit are centralized here rather than copied into every command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn

from lode.cli.render import Intent, RenderOptions, render_options_from_preset
from lode.cli.render.output import echo_json, json_err, render_message
from lode.config import Settings
from lode.embeddings.base import Embedder
from lode.index import (
    DimensionMismatchError,
    EmbedderUnavailableError,
    ModelStatus,
    SchemaVersionError,
    Store,
    StoreError,
)
from lode.ingestion.pipeline import DetectResult, SyncSummary, sync
from lode.ingestion.split import SegmentSplitter

# The index lives next to the workspace's own .lode/ directory (PLAN §3).
INDEX_DB_RELATIVE = Path(".lode") / "index.db"


def block(
    message: str,
    *,
    code: str,
    as_json: bool,
    command: str,
    intent: Intent = Intent.ERROR,
    hint: str | None = None,
) -> None:
    """Emit a blocking error (JSON or human) and exit non-zero.

    Human output renders ``message`` with ``intent`` (default ``ERROR``) and an
    optional ``hint`` with ``INFO``; JSON output joins the two into one envelope.
    """
    if as_json:
        full = f"{message}\n{hint}" if hint else message
        echo_json(json_err(command, full, code=code))
    else:
        render_message(message, intent=intent)
        if hint:
            render_message(hint, intent=Intent.INFO)
    raise typer.Exit(code=1)


def dimension_mismatch_parts(stored_dimension: int, current_dimension: int) -> tuple[str, str]:
    """Friendly dimension-mismatch message, split into the (error, hint) parts.

    Human output renders the error with ``ERROR`` and the hint with ``INFO``;
    JSON output joins the two with a newline.
    """
    error = (
        f"The current lode was driven at {stored_dimension} dimensions, but this model "
        f"yields {current_dimension}-wide nuggets — they don't sit on the same vein."
    )
    hint = (
        "Re-mine it with `lode mine --from-scratch`, or switch your embedding config "
        "back to the model/dimension that dug it."
    )
    return error, hint


def store_failure(exc: Exception, *, command: str, as_json: bool) -> None:
    """Emit a store/embedder failure through the shared blocking exit.

    Maps each failure to a stable machine-readable ``code`` (carried on the
    exception) and a friendly message, so every command surfaces errors
    symmetrically. Dimension mismatch and schema version are special-cased
    because they need a recovery hint or a re-mine prefix.
    """
    if isinstance(exc, DimensionMismatchError):
        error, hint = dimension_mismatch_parts(exc.stored_dimension, exc.current_dimension)
        block(error, code="dimension_mismatch", as_json=as_json, command=command, hint=hint)
        return
    if isinstance(exc, SchemaVersionError):
        block(f"Index needs a re-mine: {exc}", code="schema_version", as_json=as_json, command=command)
        return
    block(str(exc), code=getattr(exc, "code", "store_error"), as_json=as_json, command=command)


def open_store(
    workspace: Path,
    embedder: Embedder | None,
    *,
    command: str,
    as_json: bool,
) -> Store | None:
    """Open the workspace index, or return ``None`` if it does not exist yet.

    Only opens an existing database — creating one is ``mine``'s job. A
    missing index returns ``None`` so the caller can short-circuit (survey
    empty-classifies, prospect/dig say "run mine first"). Schema and embedder
    failures are emitted through the shared error exit.
    """
    db_path = workspace / INDEX_DB_RELATIVE
    if not db_path.exists():
        return None
    try:
        return Store(db_path, embedder)
    except (StoreError, EmbedderUnavailableError) as exc:
        store_failure(exc, command=command, as_json=as_json)
        raise typer.Exit(code=1) from exc


def render_options(settings: Settings) -> RenderOptions:
    """Resolve the configured output palette to render options.

    The command layer owns this resolution: it loads settings and hands the
    render layer a ready-made ``RenderOptions``. The render layer never reads
    config directly.

    TODO(cli-split): fold this into the command-layer palette resolution that
    will also take a future ``--palette`` flag (flag overrides config).
    """
    return render_options_from_preset(settings.output.palette)


def model_gate(
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


def sync_with_progress(
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

# Shared config option: an explicit path that skips auto-discovery (see config).
ConfigArg = Annotated[
    Path | None,
    typer.Option("--config", help="Path to a configuration file."),
]
