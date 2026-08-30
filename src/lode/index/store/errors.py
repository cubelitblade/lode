"""Exceptions raised by the index store.

Exception messages are diagnostic only (tracebacks, logs); user-facing
wording lives in the ``lode.messages`` template table, keyed by each class's
``code`` and filled from ``template_fields()``.

``EmbedderUnavailableError`` is defined in the embedding layer and re-exported
here so embedding backends can raise it without depending on the index
package; callers that catch it alongside store errors keep a single import.
``ExtensionLoadError`` follows the same pattern from the lexical layer: the
``simple`` tokenizer and the vec0 index both load shared SQLite extensions,
and the error lives where the capability is detected.
"""

from __future__ import annotations

from typing import ClassVar

from lode.embeddings.errors import EmbedderUnavailableError
from lode.lexical.errors import ExtensionLoadError

# Diagnostic message templates; never shown to users (see module docstring).
_TOKENIZER_MISMATCH_DETAIL = "index was built with tokenizer {stored!r}, but {current!r} is configured"
_DIMENSION_MISMATCH_DETAIL = "vector has {current} dimensions; the index expects {stored}"

__all__ = [
    "DimensionMismatchError",
    "EmbedderUnavailableError",
    "ExtensionLoadError",
    "MissingEmbedderError",
    "SchemaVersionError",
    "StoreError",
    "TokenizerMismatchError",
]


class StoreError(Exception):
    """Base class for index store failures."""

    # Stable machine-readable identifier for the CLI/MCP error envelope.
    code: ClassVar[str] = "store_error"

    def template_fields(self) -> dict[str, object]:
        """Fields for filling this error's ``lode.messages`` template."""
        return {}


class SchemaVersionError(StoreError):
    """The existing database was created with an incompatible schema.

    The store refuses to open; an explicit rebuild is required.
    """

    code: ClassVar[str] = "schema_version"

    def __init__(self, message: str, *, stored_version: str | None = None) -> None:
        super().__init__(message)
        self.stored_version = stored_version

    def template_fields(self) -> dict[str, object]:
        return {"stored_version": self.stored_version if self.stored_version is not None else "unknown"}


class TokenizerMismatchError(StoreError):
    """The configured tokenizer differs from the one the index was built with.

    The FTS5 table's ``tokenize=`` clause is fixed at build time, so a
    different tokenizer cannot be served from the existing index; an explicit
    rebuild is required. Carries both names so callers can present a friendly
    recovery message.
    """

    code: ClassVar[str] = "tokenizer_mismatch"

    def __init__(self, stored_tokenizer: str, current_tokenizer: str) -> None:
        super().__init__(_TOKENIZER_MISMATCH_DETAIL.format(stored=stored_tokenizer, current=current_tokenizer))
        self.stored_tokenizer = stored_tokenizer
        self.current_tokenizer = current_tokenizer

    def template_fields(self) -> dict[str, object]:
        return {"stored_tokenizer": self.stored_tokenizer, "current_tokenizer": self.current_tokenizer}


class MissingEmbedderError(StoreError):
    """Creating or rebuilding the index requires an embedder, but none was given.

    Opening an existing index never needs one; only fresh schema creation
    does, because it must ask the embedder for its vector dimension.
    """

    code: ClassVar[str] = "missing_embedder"


class DimensionMismatchError(StoreError):
    """A query or insert vector's dimension differs from the index's vec0 schema.

    The index's vec0 table is created for a fixed dimension (stored in `meta`);
    a chunk embedding a different width is incompatible and cannot be matched or
    written. Carries both widths so callers can present a friendly recovery
    message.
    """

    code: ClassVar[str] = "dimension_mismatch"

    def __init__(self, stored_dimension: int, current_dimension: int) -> None:
        super().__init__(_DIMENSION_MISMATCH_DETAIL.format(stored=stored_dimension, current=current_dimension))
        self.stored_dimension = stored_dimension
        self.current_dimension = current_dimension

    def template_fields(self) -> dict[str, object]:
        return {"stored_dimension": self.stored_dimension, "current_dimension": self.current_dimension}
