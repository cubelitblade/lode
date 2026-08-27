"""The ``assay`` sub-app: explain how a chunk is processed and scored.

Two views on one indexed chunk:

* ``assay why <digest> <query>`` — why the chunk scored as it did for a
  query: per-source scores, normalization, fusion, and the final rank,
  headed by a brief pipeline overview (details stay behind future flags).
* ``assay how <digest>`` — how the configured tokenizer actually splits the
  chunk's text at index time.

Both run a silent detection first so the stale bits are fresh before reading
chunks — these commands write ``files.status`` (they are not read-only). They
need an existing index: with none, they short-circuit with "run mine first".
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
from lode.index.search import explain
from lode.ingestion.pipeline import detect_changes
from lode.lexical import STRATEGIES, distinct_terms, tokenize_text
from lode.messages import require_error_text


def register(app: typer.Typer) -> None:
    app.add_typer(_build_assay_app(), name="assay")
    # The historical top-level alias keeps working, promoted to the whole
    # sub-app so `lode analyze why|how ...` mirrors `lode assay why|how ...`.
    app.add_typer(_build_assay_app(hidden=True), name="analyze", hidden=True)


def _build_assay_app(*, hidden: bool = False) -> typer.Typer:
    sub = typer.Typer(
        help="Explain scoring (`why`) or tokenization (`how`) for one chunk.",
        no_args_is_help=True,
    )
    sub.command("why", hidden=hidden)(why)
    sub.command("how", hidden=hidden)(how)
    return sub


def why(
    digest: Annotated[str, typer.Argument(help="Chunk digest or prefix to explain.")],
    query: Annotated[str, typer.Argument(help="Query to explain the score for.")],
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

    with store:
        # Refresh the stale bits so per-chunk provenance reflects the current
        # workspace (single dependency: only the status update, not the dirty
        # signal).
        detect_changes(store, workspace, settings.ignore.sources)
        _why(store, embedder, digest, query, settings, as_json=as_json, top_k=top_k, options=options)


def how(
    digest: Annotated[str, typer.Argument(help="Chunk digest or prefix to inspect.")],
    workspace: ProspectWorkspaceArg = Path("."),
    config: ConfigArg = None,
    palette: PaletteArg = None,
    no_color: NoColorArg = False,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Show how the configured tokenizer splits a chunk's text."""
    settings = load_settings_or_fail(config, command="assay", as_json=as_json)
    options = render_options(settings, palette, no_color)
    # No embedder needed: tokenization never touches vectors, but opening the
    # store still validates the configured tokenizer against the index.
    store = open_store(workspace, None, command="assay", as_json=as_json, tokenizer=settings.lexical.strategy)
    if store is None:
        fail_with(
            "no_index",
            command="assay",
            as_json=as_json,
            options=options,
            index_path=str(workspace / INDEX_DB_RELATIVE),
        )

    with store:
        detect_changes(store, workspace, settings.ignore.sources)
        _how(store, digest, settings.lexical.strategy, as_json=as_json, options=options)


def _resolve_chunk_rowid(
    store: Store,
    digest: str,
    *,
    as_json: bool,
    options: RenderOptions | None,
) -> int:
    """Resolve a digest (or prefix) to a single chunk rowid, or exit.

    Shared by ``why`` and ``how``; the failure shapes (invalid, unknown,
    ambiguous prefix) are identical for both views.
    """
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
    return rowids[0]


def _why(
    store: Store,
    embedder: Embedder,
    digest: str,
    query: str,
    settings: Settings,
    *,
    as_json: bool,
    top_k: int | None,
    options: RenderOptions | None,
) -> None:
    if not query.strip():
        fail_with("invalid_query", command="assay", as_json=as_json, options=options)

    rowid = _resolve_chunk_rowid(store, digest, as_json=as_json, options=options)

    resolved_top_k = top_k if top_k is not None else settings.retrieval.top_k
    try:
        explanation = explain(
            store,
            embedder,
            query,
            rowid,
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


def _how(
    store: Store,
    digest: str,
    strategy_name: str,
    *,
    as_json: bool,
    options: RenderOptions | None,
) -> None:
    rowid = _resolve_chunk_rowid(store, digest, as_json=as_json, options=options)
    chunk = store.get_chunks([rowid]).get(rowid)
    if chunk is None:  # pragma: no cover - defensive: the rowid was just resolved
        fail_with("not_found", command="assay", as_json=as_json, options=options, digest=digest)

    strategy = STRATEGIES[strategy_name]
    tokens = tokenize_text(strategy, chunk.text)
    terms = strategy.interpret(tokens)

    if as_json:
        echo_json(json_ok("assay", **_how_json(chunk, strategy, tokens, terms)))
    else:
        _cli.render_how(
            chunk,
            tokenizer=strategy.name,
            tokenize_clause=strategy.tokenize_clause,
            terms=terms,
            options=options,
        )


def _how_json(chunk: Any, strategy: Any, tokens: list[str], terms: list[Any]) -> dict[str, Any]:
    """JSON payload for a tokenization view.

    ``tokens`` is the raw index-side stream (machine-readable contract);
    ``terms`` is the structured interpretation with pinyin variants folded in.
    """
    distinct = distinct_terms(terms)
    return {
        "digest": chunk.digest,
        "paths": [{"path": ref.path, "state": ref.status.value} for ref in chunk.refs],
        "heading": chunk.heading,
        "page": chunk.page,
        "seq": chunk.seq,
        "tokenizer": {"strategy": strategy.name, "tokenize_clause": strategy.tokenize_clause},
        "text": chunk.text,
        "char_count": len(chunk.text),
        "token_count": len(tokens),
        "term_count": len(distinct),
        "tokens": tokens,
        "terms": [{"surface": term.surface, "variants": list(term.variants)} for term in distinct],
    }


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


def _explanation_json(explanation: Any) -> dict[str, Any]:
    """JSON payload for a score explanation."""
    chunk = explanation.chunk
    plan = explanation.plan
    norm_params = dict(plan.norm.params) if plan.norm is not None else {}
    return {
        "digest": chunk.digest,
        "paths": [{"path": ref.path, "state": ref.status.value} for ref in chunk.refs],
        "heading": chunk.heading,
        "page": chunk.page,
        "seq": chunk.seq,
        "sources": {
            name: {
                "status": source.status.value,
                "pool_size": source.pool_size,
                "raw": source.raw_score,
                "prepared": source.prepared_score,
                "pool_rank": source.pool_rank,
            }
            for name, source in explanation.sources.items()
        },
        "norm": {"name": plan.norm.name if plan.norm is not None else None, "params": norm_params},
        "fusion": {"name": plan.fusion.name, "params": dict(plan.fusion.params)},
        "combined": explanation.combined,
        "rank": explanation.rank,
        "in_results": explanation.in_results,
        "top_k": explanation.top_k,
    }
