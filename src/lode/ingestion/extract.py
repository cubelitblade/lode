"""Text extraction: file bytes + suffix -> plain text.

The extractor owns the supported-format table. Unsupported files return
None and are skipped by the pipeline. MVP: plain text and markdown; office
formats (docx/pdf/xlsx) land in M3 per PLAN.

Decoding is best-effort with a safe fallback chain: UTF-8 (with BOM) first,
then UTF-16, then Latin-1 — which never fails, so extraction always yields
some text rather than raising on an exotic encoding.
"""

from __future__ import annotations

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})

# Decoding fallback chain, most specific first. UTF-16 is only attempted
# when a BOM is present: Python's utf-16 codec happily decodes arbitrary
# even-length bytes as garbage (native byte order), which would swallow
# Latin-1 text intended for the last fallback.
_DECODINGS = ("utf-8-sig", "utf-16", "latin-1")

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


def is_supported(suffix: str) -> bool:
    """Whether the file extension is extractable."""
    return suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(data: bytes, suffix: str) -> str | None:
    """Plain text for a supported file, or None for unsupported formats."""
    if not is_supported(suffix):
        return None
    return decode_text(data)


def decode_text(data: bytes) -> str:
    """Decode bytes to text, falling back through encodings that cannot fail."""
    for encoding in _DECODINGS:
        if encoding == "utf-16" and not data.startswith(_UTF16_BOMS):
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 never raises; unreachable in practice, kept for exhaustiveness.
    return data.decode("latin-1", errors="replace")
