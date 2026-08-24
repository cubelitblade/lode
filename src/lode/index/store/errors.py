"""Exceptions raised by the index store."""

from __future__ import annotations


class StoreError(Exception):
    """Base class for index store failures."""


class SchemaVersionError(StoreError):
    """The existing database was created with an incompatible schema.

    The store refuses to open; an explicit rebuild is required.
    """


class EmbedderUnavailableError(StoreError):
    """The embedder could not provide metadata needed to (re)build the index."""


class DimensionMismatchError(StoreError):
    """A query or insert vector's dimension differs from the index's vec0 schema.

    The index's vec0 table is created for a fixed dimension (stored in `meta`);
    a chunk embedding a different width is incompatible and cannot be matched or
    written. Carries both widths so callers can present a friendly recovery
    message.
    """

    def __init__(self, stored_dimension: int, current_dimension: int, *, operation: str) -> None:
        super().__init__(f"{operation} vector has {current_dimension} dimensions; the index expects {stored_dimension}")
        self.stored_dimension = stored_dimension
        self.current_dimension = current_dimension
        self.operation = operation
