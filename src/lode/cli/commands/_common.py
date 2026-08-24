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
from typing import Annotated, Literal

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
) -> None:
    """Emit a blocking error (JSON or human) and exit non-zero.

    Human output renders ``message`` with ``intent`` and an optional ``hint``
    with ``INFO``; JSON output joins the two into one envelope.
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
    """Friendly dimension-mismatch message, split into (error, hint) parts."""
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
    exception) and a friendly message. Dimension mismatch and schema version are
    special-cased for a recovery hint or a re-mine prefix.
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

    Only opens an existing database — creating one is ``mine``'s job. A missing
    index returns ``None`` so the caller can short-circuit. Schema and embedder
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


def model_gate(
    store: Store,
    embedder: Embedder,
    *,
    from_scratch: bool = False,
    as_json: bool = False,
    command: str = "mine",
) -> None:
    """Enforce the model/dimension-consistency contract for mine/prospect.

    A mismatch means the index was built with a different model or dimension:
    querying is refused and updating without a from-scratch re-mine would keep
    incompatible vectors. UNKNOWN (embedding endpoint down) does not block.
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
        # block search, which can still serve cached data.
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
