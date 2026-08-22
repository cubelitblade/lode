"""lode CLI: survey / mine / prospect (aliases: status / index / search).

The mining metaphor carries the narrative (survey = detect, mine = embed,
prospect = search); the aliases keep the interface approachable for
practical use. MCP tools (index_status/reindex/search) are a thin layer on
the same functions — CLI first, MCP later (PLAN M1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lode.config import build_embedder, load_settings
from lode.index import EmbedderUnavailableError, ModelStatus, SchemaVersionError, Store
from lode.index.search import search
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


@app.command("survey")
@app.command("status", hidden=True)
def survey(
    workspace: WorkspaceArg = Path("."),
) -> None:
    """Detect workspace changes against the index; mark changed files stale.

    No embedding endpoint is needed — pure file comparison.
    """
    settings = load_settings()
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        typer.echo(f"Index needs a rebuild: {exc}")
        raise typer.Exit(code=1) from exc
    with store:
        result = survey_workspace(store, workspace, settings.ignore.files)
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
) -> None:
    """Embed changed/new files and update the index (requires an embedding endpoint).

    Use `--rebuild` after changing the embedding model or if the index
    schema is incompatible.
    """
    settings = load_settings()
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
                RecursiveSegmentSplitter(),
                settings.ignore.files,
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
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="Number of results to return.")] = 10,
) -> None:
    """Search the index and show results with provenance (file + heading)."""
    settings = load_settings()
    embedder = build_embedder(settings.embedding)
    try:
        store = Store(workspace / INDEX_DB_RELATIVE, embedder)
    except SchemaVersionError as exc:
        typer.echo(f"Index needs a rebuild: {exc}")
        raise typer.Exit(code=1) from exc

    _model_gate(store)
    with store:
        hits = search(
            store,
            embedder,
            query,
            dense_weight=settings.retrieval.dense_weight,
            sparse_weight=settings.retrieval.sparse_weight,
            top_k=top_k,
        )
        if not hits:
            typer.echo("Dry hole: nothing matched.")
            return
        for index, hit in enumerate(hits, start=1):
            stale_tag = " [stale]" if hit.stale else ""
            heading = f" > {hit.heading}" if hit.heading else ""
            typer.echo(f"{index}. [{hit.score:.3f}] {hit.path}{heading}{stale_tag}")
            snippet = hit.text.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            typer.echo(f"   {snippet}")


if __name__ == "__main__":
    app()
