"""Exceptions raised by the ingestion layer.

``ExtractionError`` is the single domain error the pipeline treats as a
per-file, recoverable failure. It wraps the (unbounded) set of exceptions
third-party parsers (python-docx, PyMuPDF) can raise on untrusted file
bytes, so ``sync`` can catch one type instead of a broad ``Exception``.
"""

from __future__ import annotations


class ExtractionError(Exception):
    """A file's bytes could not be parsed into segments.

    Raised by ``extract_document`` when a structured format (docx/pdf) fails
    to parse. The original exception is preserved via ``__cause__``. This is
    a *domain* error: the file is bad, not the code — the pipeline flags the
    file stale and moves on.
    """
