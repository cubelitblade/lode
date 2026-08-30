"""Index header inspection and wholesale replacement.

These entry points complement ``Store`` for the CLI's exemption flow.
``read_index_meta`` and ``check_index_compatibility`` classify an existing database
against the current configuration *before* a ``Store`` is constructed, so
every mismatch (schema version, model, dimension, tokenizer) surfaces
through one uniform exit instead of leaking as constructor failures.
``reset_index`` implements ``mine --from-scratch``'s permission to destroy
the index: embedder metadata is resolved first so an unreachable endpoint
changes nothing, the old database is snapshotted to ``<db_path>.bak``, then
removed so the next ``Store`` construction starts fresh.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# The sqlite_vec package ships no type stubs; the ignore is scoped to this
# import line and should be revisited after a dependency upgrade.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]

from lode.embeddings.base import Embedder
from lode.embeddings.errors import EmbedderUnavailableError
from lode.index.store.schema import SCHEMA_VERSION
from lode.lexical import STRATEGIES

# SQLite does not accept parameter binding for DDL, so the VACUUM INTO
# target is a quoted literal; escape any embedded quotes defensively.
_VACUUM_INTO = "VACUUM INTO '{}'"

# Databases predating the tokenizer key were built with the historical
# default; mirror Store's legacy fallback so both paths agree.
_LEGACY_TOKENIZER = "unicode61"


@dataclass(frozen=True, slots=True)
class IndexMeta:
    """The ``meta`` header of an existing index database."""

    schema_version: str | None
    model_id: str | None
    dimension: str | None
    tokenizer: str | None


@dataclass(frozen=True, slots=True)
class IndexIssue:
    """One way an existing index is incompatible with the current configuration.

    ``code`` doubles as the ``lode.messages`` template key and the JSON
    envelope code; ``fields`` fills that template.
    """

    code: str
    fields: dict[str, object]


def read_index_meta(db_path: Path) -> IndexMeta | None:
    """Read an index's ``meta`` header without constructing a ``Store``.

    Returns ``None`` when the file does not exist, has no ``meta`` table, or
    cannot be read as SQLite at all — callers classify that as a schema
    incompatibility rather than crashing. Never touches the embedder, and
    never creates the file when it is missing.
    """
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return None
    try:
        # The schema declares vec0/FTS5 virtual tables; loading the extension
        # keeps even eager schema parsing working on any connection.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()
        if has_meta is None:
            return None
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    values = {str(key): str(value) for key, value in rows}
    return IndexMeta(
        schema_version=values.get("schema_version"),
        model_id=values.get("model_id"),
        dimension=values.get("dimension"),
        tokenizer=values.get("tokenizer"),
    )


def check_index_compatibility(
    meta: IndexMeta | None,
    *,
    embedder: Embedder | None,
    tokenizer: str,
) -> tuple[IndexIssue, ...]:
    """Classify an index header against the current configuration.

    Returns every incompatibility in fixed priority order (schema version,
    model, dimension, tokenizer); callers report the first. An unreadable
    header counts as a schema incompatibility. Embedder metadata that cannot
    be resolved (endpoint down, no embedder) is skipped fault-tolerantly, so
    merely serving an index never blocks on the embedding endpoint.
    """
    if meta is None:
        return (IndexIssue("schema_version", {"stored_version": "unknown"}),)

    if meta.schema_version != str(SCHEMA_VERSION):
        # Later checks would read another era's meta layout; report this alone.
        stored = meta.schema_version if meta.schema_version is not None else "unknown"
        return (IndexIssue("schema_version", {"stored_version": stored}),)

    issues: list[IndexIssue] = []

    current_model_id = _embedder_model_id(embedder)
    if meta.model_id is not None and current_model_id is not None and current_model_id != meta.model_id:
        issues.append(IndexIssue("model_mismatch", {"stored_model_id": meta.model_id}))

    stored_dimension = _optional_int(meta.dimension)
    current_dimension = _embedder_dimension(embedder)
    if stored_dimension is not None and current_dimension is not None and current_dimension != stored_dimension:
        issues.append(
            IndexIssue(
                "dimension_mismatch",
                {"stored_dimension": stored_dimension, "current_dimension": current_dimension},
            )
        )

    try:
        current_clause = STRATEGIES[tokenizer].tokenize_clause
    except KeyError as exc:
        raise ValueError(f"unknown tokenizer {tokenizer!r}; choose from {', '.join(STRATEGIES)}") from exc
    stored_tokenizer = meta.tokenizer or _LEGACY_TOKENIZER
    if stored_tokenizer != current_clause:
        issues.append(
            IndexIssue(
                "tokenizer_mismatch",
                {"stored_tokenizer": stored_tokenizer, "current_tokenizer": current_clause},
            )
        )

    return tuple(issues)


def reset_index(db_path: Path, embedder: Embedder, tokenizer: str) -> None:
    """Snapshot the index to ``<db_path>.bak`` and remove it (WAL sidecars too).

    ``mine --from-scratch``'s destructive step. Both the embedder metadata and
    the tokenizer name are validated *before* anything is touched, so a failed
    reset leaves the existing index fully intact. The next ``Store``
    construction on the same path starts from a fresh schema carrying the
    current configuration.

    **Atomicity** (crash-safe order):
    1. VACUUM INTO ``<db>.bak.tmp`` — if this fails the original db and any
       previous ``.bak`` are untouched.
    2. ``.bak.tmp`` → ``.bak`` rename — atomic on the same filesystem, so a
       crash during rename leaves either the old or the new ``.bak``, never
       a half-written file.
    3. Remove the original db and its WAL/SHM sidecars.
    """
    if tokenizer not in STRATEGIES:
        raise ValueError(f"unknown tokenizer {tokenizer!r}; choose from {', '.join(STRATEGIES)}")
    try:
        embedder.dimension  # noqa: B018
        embedder.model_id  # noqa: B018
    except Exception as exc:
        raise EmbedderUnavailableError(f"cannot reset index: could not determine embedding metadata: {exc}") from exc

    tmp_path = Path(f"{db_path}.bak.tmp")
    final_path = Path(f"{db_path}.bak")
    tmp_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        quoted = tmp_path.as_posix().replace("'", "''")
        conn.execute(_VACUUM_INTO.format(quoted))
    finally:
        conn.close()
    # fsync the backup so it is durable before we touch the original.
    # The handle must be opened for writing: on Windows os.fsync (_commit)
    # fails with EBADF on a read-only descriptor.
    with open(tmp_path, "rb+") as f:
        os.fsync(f.fileno())
    # Atomic rename: after this point the backup is committed.
    tmp_path.replace(final_path)
    # Remove the original database — WAL and SHM sidecars first.
    for suffix in ("-wal", "-shm", ""):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def _embedder_model_id(embedder: Embedder | None) -> str | None:
    if embedder is None:
        return None
    try:
        return embedder.model_id
    except Exception:
        return None


def _embedder_dimension(embedder: Embedder | None) -> int | None:
    if embedder is None:
        return None
    try:
        return embedder.dimension
    except Exception:
        return None


def _optional_int(raw: str | None) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None
