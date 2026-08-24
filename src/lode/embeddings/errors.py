"""Exceptions raised by the embedding layer.

``EmbedderUnavailableError`` lives here (not in the index layer) so that
embedding backends can raise it without depending on the index package.
The index store re-exports it for callers that catch it alongside store
errors.
"""

from __future__ import annotations


class EmbedderUnavailableError(Exception):
    """The embedder could not provide metadata or embeddings.

    Raised when the embedding endpoint is unreachable, times out, or returns
    an error after retries are exhausted. Carries the underlying cause via
    ``__cause__`` so callers can log or present it.
    """