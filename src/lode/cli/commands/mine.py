"""The ``mine`` command: embed and index new or changed files.

This is the only command that creates the index database (and thus the only
one that may reach the embedding endpoint for the vector dimension). With no
index yet it classifies against an empty snapshot first: if there is nothing
to embed it reports ``Nothing to do.`` without creating a database or
touching the embedder; otherwise it creates the index and syncs.

``--from-scratch`` is the single mismatch exemption: it snapshots the old
database to ``<db>.bak``, removes it, and re-mines every file under the
current configuration — no mismatched index is ever opened.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

import lode.cli as _cli
from lode.cli.commands._common import (
    INDEX_DB_RELATIVE,
    ConfigArg,
    NoColorArg,
    PaletteArg,
    load_settings_or_fail,
    open_store,
    render_options,
    store_failure,
    sync_with_progress,
    workspace_from,
)
from lode.cli.render import RenderOptions
from lode.cli.render.output import echo_json, json_ok
from lode.config import Settings
from lode.embeddings.base import Embedder
from lode.index import EmbedderUnavailableError, Store, StoreError, reset_index
from lode.ingestion.pipeline import (
    DetectResult,
    SyncSummary,
    classify,
    detect_changes,
    sync,
)
from lode.ingestion.split import RecursiveSegmentSplitter, SegmentSplitter


def register(app: typer.Typer) -> None:
    app.command("mine")(mine)
    app.command("index", hidden=True)(mine)


def mine(
    ctx: typer.Context,
    from_scratch: bool = typer.Option(
        False,
        "--from-scratch",
        help="Discard the existing index and create a new one from scratch.",
    ),
    config: ConfigArg = None,
    palette: PaletteArg = None,
    no_color: NoColorArg = False,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Embeds and indexes new or changed files. An embedding endpoint must be configured.

    Use `--from-scratch` when the index no longer matches your configuration
    (embedding model, dimension, tokenizer) or its schema is incompatible;
    it snapshots the old index to `<index>.bak` and re-mines everything.
    """
    workspace = workspace_from(ctx)
    settings = load_settings_or_fail(config, command="mine", as_json=as_json)
    embedder = _cli.build_embedder(settings.embedding)
    options = render_options(settings, palette, no_color)
    db_path = workspace / INDEX_DB_RELATIVE
    if from_scratch and db_path.exists():
        # The one exemption path: destroy the index (after snapshotting it)
        # so every file is re-mined under the current configuration. A failed
        # reset leaves the old index intact.
        try:
            reset_index(db_path, embedder, tokenizer=settings.lexical.strategy)
        except (StoreError, EmbedderUnavailableError) as exc:
            store_failure(exc, command="mine", as_json=as_json)

    store = open_store(workspace, embedder, command="mine", as_json=as_json, tokenizer=settings.lexical.strategy)
    if store is None:
        # No index yet: classify against an empty snapshot. If there is
        # nothing to embed, report Nothing to do. without creating a database
        # or touching the embedder; otherwise create the index and sync.
        detect = classify({}, workspace, settings.ignore.sources)
        if detect.pending == 0:
            result = SyncSummary()
            result.unchanged = detect.unchanged
            result.skipped = detect.skipped
            _emit_mine(workspace, result, from_scratch, as_json, options)
            return
        try:
            store = Store(db_path, embedder, tokenizer=settings.lexical.strategy)
        except (StoreError, EmbedderUnavailableError) as exc:
            store_failure(exc, command="mine", as_json=as_json)
        # A fresh database has no stored model to gate against.
        with store:
            splitter = _splitter(settings)
            result = _run_sync(store, workspace, embedder, splitter, detect, as_json, no_color=options.no_color)
        _emit_mine(workspace, result, from_scratch, as_json, options)
        return

    with store:
        splitter = _splitter(settings)
        detect = detect_changes(store, workspace, settings.ignore.sources)
        result = _run_sync(store, workspace, embedder, splitter, detect, as_json, no_color=options.no_color)
    _emit_mine(workspace, result, from_scratch, as_json, options)


def _splitter(settings: Settings) -> SegmentSplitter:
    """Build the segment splitter from chunking settings."""
    return RecursiveSegmentSplitter(
        chunk_size=settings.chunking.size,
        chunk_overlap=settings.chunking.overlap,
    )


def _run_sync(
    store: Store,
    workspace: Path,
    embedder: Embedder,
    splitter: SegmentSplitter,
    detect: DetectResult,
    as_json: bool,
    no_color: bool | None = None,
) -> SyncSummary:
    """Run sync, with a progress bar on a TTY unless JSON output is requested."""
    console = Console(no_color=no_color)
    if detect.pending == 0:
        # Nothing to embed or prune; skip the progress bar entirely.
        result = SyncSummary()
        result.unchanged = detect.unchanged
        result.skipped = detect.skipped
        return result
    if as_json:
        return sync(store, workspace, embedder, splitter, detect=detect)
    if console.is_terminal:
        return sync_with_progress(console, store, workspace, embedder, splitter, detect)
    return sync(store, workspace, embedder, splitter, detect=detect)


def _emit_mine(
    workspace: Path,
    result: SyncSummary,
    from_scratch: bool,
    as_json: bool,
    options: RenderOptions,
) -> None:
    """Emit the mine result as JSON or human-readable output."""
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
                    "renamed": result.renamed,
                    "skipped": result.skipped,
                },
                paths={
                    "added": result.added_files,
                    "updated": result.updated_files,
                    "removed": result.removed_files,
                    "renamed": [{"from": old, "to": new} for old, new in result.renamed_files],
                },
                failed=[{"path": failure.path, "error": failure.error} for failure in result.failed],
            )
        )
    else:
        _cli.render_mine(workspace, result, console=Console(no_color=options.no_color), options=options)
