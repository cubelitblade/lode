"""Content addressing primitives for chunk deduplication.

A chunk's identity is derived from its normalized text, so unchanged
content keeps the same `chunk_digest` no matter where it appears or how the
surrounding file changed (see PLAN: content-addressed storage, D4/D6).
The normalization rule below is a stability contract: changing it
invalidates every previously stored digest, so it is locked by tests.
"""

from __future__ import annotations

import re

import blake3

# Algorithm tag embedded in every digest, so future algorithm switches
# (e.g. a different hash) can be detected without a schema migration.
_DIGEST_PREFIX = "blake3:"

# Collapse runs of horizontal whitespace inside a line. Newlines are kept:
# chunk boundaries and paragraph structure must survive normalization.
_INLINE_WS = re.compile(r"[ \t\f\v]+")


def normalize(text: str) -> str:
    """Stable canonical form of chunk text.

    Rules (do not change without a full reindex):
      - strip leading/trailing whitespace (including newlines)
      - collapse runs of spaces/tabs/form-feed/vertical-tab to a single space
      - keep internal newlines, paragraph breaks, and case intact
    """
    return _INLINE_WS.sub(" ", text).strip()


def chunk_digest(text: str) -> str:
    """Content address: ``blake3:<hex>`` of the normalized text."""
    digest = blake3.blake3(normalize(text).encode("utf-8")).hexdigest()
    return f"{_DIGEST_PREFIX}{digest}"


def file_digest(data: bytes) -> str:
    """File-level content address: ``blake3:<hex>`` of the raw bytes.

    Used for rename detection and change attribution (PLAN D6); unlike
    ``chunk_digest`` it hashes the raw bytes, not a normalized form.
    """
    return f"{_DIGEST_PREFIX}{blake3.blake3(data).hexdigest()}"
