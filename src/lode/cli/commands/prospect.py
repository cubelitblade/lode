"""The ``prospect`` command: search the index and show results with sources.

Runs a silent detection first so the stale bits are fresh before search
reads them — this command writes ``files.status`` (it is not read-only). It
needs an existing index: with none, it short-circuits with "run mine first".
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import lode.cli as _cli
from lode.cli.commands._common import (
    ConfigArg,
    NoColorArg,
    PaletteArg,
    ProspectWorkspaceArg,
    dimension_mismatch_parts,
    model_gate,
    open_store,
    render_options,
)
from lode.cli.render import Intent
from lode.cli.render.output import echo_json, json_err, json_ok, preview, render_message
from lode.config import build_plan, load_settings
from lode.index import DimensionMismatchError
from lode.index.search import ProspectResult, search
from lode.ingestion.pipeline import detect_changes


def register(app: typer.Typer) -> None:
    app.command("prospect")(prospect)
    app.command("search", hidden=True)(prospect)


def prospect(
    query: Annotated[str, typer.Argument(help="Query to search for.")],
    workspace: ProspectWorkspaceArg = Path("."),
    top_k: Annotated[int | None, typer.Option("--top-k", min=1, help="Maximum number of results to return.")] = None,
    config: ConfigArg = None,
    palette: PaletteArg = None,
    no_color: NoColorArg = False,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Search the index and show results with source information.

    Runs a silent detection first so the stale bits are fresh before search
    reads them — this command writes ``files.status`` (it is not read-only).
    """
    settings = load_settings(config)
    embedder = _cli.build_embedder(settings.embedding)
    options = render_options(settings, palette, no_color)
    store = open_store(workspace, embedder, command="prospect", as_json=as_json, tokenizer=settings.lexical.strategy)
    if store is None:
        message = f"Dry hole: no index at {workspace / '.lode' / 'index.db'}; run `lode mine` first."
        if as_json:
            echo_json(json_err("prospect", message, code="no_index"))
        else:
            render_message(message, intent=Intent.ERROR)
        raise typer.Exit(code=1)

    model_gate(store, embedder, as_json=as_json, command="prospect")
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
                plan=build_plan(settings.norm, settings.fusion),
                top_k=top_k,
            )
        except ValueError as exc:
            if as_json:
                echo_json(json_err("prospect", str(exc), code="invalid_query"))
            else:
                render_message(str(exc), intent=Intent.ERROR)
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
            _cli.render_prospect(result, options=options)
