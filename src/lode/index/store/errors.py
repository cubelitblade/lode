"""Exceptions raised by the index store.

``EmbedderUnavailableError`` is defined in the embedding layer and re-exported
here so embedding backends can raise it without depending on the index
package; callers that catch it alongside store errors keep a single import.
"""

from __future__ import annotations

from typing import ClassVar

from lode.embeddings.errors import EmbedderUnavailableError

__all__ = [
    "DimensionMismatchError",
    "EmbedderUnavailableError",
    "MissingEmbedderError",
    "SchemaVersionError",
    "StoreError",
    "TokenizerMismatchError",
]


class StoreError(Exception):
    """Base class for index store failures."""

    # Stable machine-readable identifier for the CLI/MCP error envelope.
    code: ClassVar[str] = "store_error"


class SchemaVersionError(StoreError):
    """The existing database was created with an incompatible schema.

    The store refuses to open; an explicit rebuild is required.
    """

    code: ClassVar[str] = "schema_version"


class TokenizerMismatchError(StoreError):
    """The configured tokenizer differs from the one the index was built with.

    The FTS5 table's ``tokenize=`` clause is fixed at build time, so a
    different tokenizer cannot be served from the existing index; an explicit
    rebuild is required. Carries both names so callers can present a friendly
    recovery message.
    """

    code: ClassVar[str] = "tokenizer_mismatch"

    def __init__(self, stored_tokenizer: str, current_tokenizer: str) -> None:
        super().__init__(
            f"index was built with tokenizer {stored_tokenizer!r}, but {current_tokenizer!r} is configured"
        )
        self.stored_tokenizer = stored_tokenizer
        self.current_tokenizer = current_tokenizer


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

    def __init__(self, stored_dimension: int, current_dimension: int, *, operation: str) -> None:
        super().__init__(f"{operation} vector has {current_dimension} dimensions; the index expects {stored_dimension}")
        self.stored_dimension = stored_dimension
        self.current_dimension = current_dimension
        self.operation = operation
