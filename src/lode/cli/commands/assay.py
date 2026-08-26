"""The ``assay`` command: explain why a chunk scored as it did for a query.

Runs a silent detection first so the stale bits are fresh before reading
chunks — this command writes ``files.status`` (it is not read-only). It
needs an existing index: with none, it short-circuits with "run mine first".
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

import lode.cli as _cli
from lode.cli.commands._common import (
    DIGEST_PATTERN,
    INDEX_DB_RELATIVE,
    ConfigArg,
    NoColorArg,
    PaletteArg,
    ProspectWorkspaceArg,
    fail_with,
    load_settings_or_fail,
    model_gate,
    normalize_digest,
    open_store,
    render_options,
    store_failure,
)
from lode.cli.render import Intent, RenderOptions
from lode.cli.render.output import echo_json, json_err, json_ok, render_message
from lode.config import Settings, build_plan
from lode.embeddings.base import Embedder
from lode.index import DimensionMismatchError, Store
from lode.index.search import ScoreExplanation, explain
from lode.ingestion.pipeline import detect_changes
from lode.messages import require_error_text


def register(app: typer.Typer) -> None:
    app.command("assay")(assay)
    app.command("analyze", hidden=True)(assay)


def assay(
    query: Annotated[str, typer.Argument(help="Query to explain the score for.")],
    digest: Annotated[str, typer.Argument(help="Chunk digest or prefix to explain.")],
    workspace: ProspectWorkspaceArg = Path("."),
    top_k: Annotated[int | None, typer.Option("--top-k", min=1, help="Maximum number of results to return.")] = None,
    config: ConfigArg = None,
    palette: PaletteArg = None,
    no_color: NoColorArg = False,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Explain why a chunk scored as it did for a query."""
    settings = load_settings_or_fail(config, command="assay", as_json=as_json)
    embedder = _cli.build_embedder(settings.embedding)
    options = render_options(settings, palette, no_color)
    store = open_store(workspace, embedder, command="assay", as_json=as_json, tokenizer=settings.lexical.strategy)
    if store is None:
        fail_with(
            "no_index",
            command="assay",
            as_json=as_json,
            options=options,
            index_path=str(workspace / INDEX_DB_RELATIVE),
        )

    model_gate(store, embedder, as_json=as_json, command="assay")
    with store:
        # Refresh the stale bits so per-chunk provenance reflects the current
        # workspace (single dependency: only the status update, not the dirty
        # signal).
        detect_changes(store, workspace, settings.ignore.sources)
        _assay(store, embedder, query, digest, settings, as_json=as_json, top_k=top_k, options=options)


def _assay(
    store: Store,
    embedder: Embedder,
    query: str,
    digest: str,
    settings: Settings,
    *,
    as_json: bool,
    top_k: int | None,
    options: RenderOptions | None,
) -> None:
    if not query.strip():
        fail_with("invalid_query", command="assay", as_json=as_json, options=options)

    token = normalize_digest(digest)
    if not DIGEST_PATTERN.fullmatch(token):
        fail_with("invalid_digest", command="assay", as_json=as_json, options=options, digest=digest)

    rowids = store.find_chunk_rowids(token)
    if not rowids:
        fail_with("not_found", command="assay", as_json=as_json, options=options, digest=digest)
    if len(rowids) > 1:
        # Structured candidates: the JSON envelope carries them as data, the
        # human output lists them as muted lines under the message.
        text = require_error_text("ambiguous", digest=digest, count=len(rowids))
        if as_json:
            echo_json(
                json_err(
                    "assay",
                    f"{text.error}\n{text.hint}",
                    code="ambiguous",
                    candidates=[_candidate_json(store, rowid) for rowid in rowids],
                )
            )
        else:
            render_message(text.error, intent=Intent.ERROR, options=options)
            render_message(text.hint, intent=Intent.INFO, options=options)
            for rowid in rowids:
                render_message(_candidate_line(store, rowid), intent=Intent.MUTED, options=options)
        raise typer.Exit(code=1)

    resolved_top_k = top_k if top_k is not None else settings.retrieval.top_k
    try:
        explanation = explain(
            store,
            embedder,
            query,
            rowids[0],
            plan=build_plan(settings.norm, settings.fusion),
            top_k=resolved_top_k,
        )
    except ValueError as exc:
        if as_json:
            echo_json(json_err("assay", str(exc), code="invalid_query"))
        else:
            render_message(str(exc), intent=Intent.ERROR, options=options)
        raise typer.Exit(code=1) from exc
    except DimensionMismatchError as exc:
        # Fallback for a dimension mismatch the gate could not detect: surface
        # the friendly recovery message instead of a raw sqlite traceback.
        store_failure(exc, command="assay", as_json=as_json)

    if as_json:
        echo_json(
            json_ok(
                "assay",
                query=query,
                digest=digest,
                top_k=resolved_top_k,
                explanation=_explanation_json(explanation),
            )
        )
    else:
        _cli.render_assay(explanation, query=query, options=options)


def _candidate_line(store: Store, rowid: int) -> str:
    """One-line provenance for an ambiguous-digest candidate."""
    chunk = store.get_chunks([rowid]).get(rowid)
    if chunk is None:
        return f"  #{rowid}"
    short_id = chunk.digest.removeprefix("blake3:")[:12]
    heading = f" > {chunk.heading}" if chunk.heading else ""
    return f"  #{short_id} {chunk.primary.path}{heading}"


def _candidate_json(store: Store, rowid: int) -> dict[str, Any]:
    """JSON payload for one ambiguous-digest candidate."""
    chunk = store.get_chunks([rowid]).get(rowid)
    if chunk is None:
        return {"rowid": rowid}
    return {
        "rowid": rowid,
        "digest": chunk.digest,
        "paths": [{"path": ref.path, "state": ref.status.value} for ref in chunk.refs],
        "heading": chunk.heading,
        "page": chunk.page,
    }


def _explanation_json(explanation: ScoreExplanation) -> dict[str, Any]:
    """JSON payload for a score explanation."""
    chunk = explanation.chunk
    plan = explanation.plan
    return {
        "digest": chunk.digest,
        "paths": [{"path": ref.path, "state": ref.status.value} for ref in chunk.refs],
        "heading": chunk.heading,
        "page": chunk.page,
        "seq": chunk.seq,
        "semantic": {
            "raw": explanation.semantic_raw,
            "prepared": explanation.semantic_prepared,
            "pool_rank": explanation.semantic_pool_rank,
            "pool_size": explanation.semantic_pool_size,
        },
        "lexical": {
            "raw": explanation.lexical_raw,
            "prepared": explanation.lexical_prepared,
            "pool_rank": explanation.lexical_pool_rank,
            "pool_size": explanation.lexical_pool_size,
        },
        "norm": {"name": plan.norm.name if plan.norm is not None else None},
        "fusion": {"name": plan.fusion.name},
        "combined": explanation.combined,
        "rank": explanation.rank,
        "in_results": explanation.in_results,
        "top_k": explanation.top_k,
    }
