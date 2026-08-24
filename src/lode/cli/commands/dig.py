"""The ``dig`` command: fetch a chunk's full text by its digest.

Runs a silent detection first so the stale bits are fresh before reading
chunks — this command writes ``files.status`` (it is not read-only). It
needs an existing index: with none, it short-circuits with "run mine first".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

import typer

import lode.cli as _cli
from lode.cli.commands._common import (
    ConfigArg,
    DigWorkspaceArg,
    open_store,
    render_options,
)
from lode.cli.render import RenderOptions
from lode.cli.render.output import echo_json, json_err, json_ok
from lode.config import load_settings
from lode.index import ChunkWithPath, Store
from lode.ingestion.pipeline import detect_changes

# Hex hexdigest bodies (BLAKE3 produces lowercase hex); accept uppercase too.
_DIGEST_PATTERN = re.compile(r"[0-9a-f]+", re.IGNORECASE)


def register(app: typer.Typer) -> None:
    app.command("dig")(dig)
    app.command("get", hidden=True)(dig)


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
    settings = load_settings(config)
    options = render_options(settings)
    store = open_store(workspace, None, command="dig", as_json=as_json)
    if store is None:
        message = f"Dry hole: no index at {workspace / '.lode' / 'index.db'}; run `lode mine` first."
        if as_json:
            echo_json(json_err("dig", message, code="no_index"))
        else:
            typer.echo(message)
        raise typer.Exit(code=1)
    with store:
        # Refresh the stale bits so per-chunk provenance reflects the current
        # workspace (single dependency: only the status update, not the dirty
        # signal).
        detect_changes(store, workspace, settings.ignore.sources)
        _dig(store, digest, as_json=as_json, radius=radius or 0, options=options)


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
        _cli.render_dig(
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
