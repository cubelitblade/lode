"""User-facing message templates.

Templates are keyed by a stable code — a ``StoreError.code`` for domain
failures, or a CLI failure code for fixed-shape command exits — and rendered
with ``str.format`` from caller-supplied fields (store exceptions expose
theirs through ``StoreError.template_fields()``). This module is the single
source of user-facing wording — shared by the CLI and the future MCP layer;
exception messages themselves are diagnostic only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ErrorText:
    """A rendered error line plus an optional recovery hint."""

    error: str
    hint: str = ""


_TEMPLATES: Final[Mapping[str, ErrorText]] = {
    "dimension_mismatch": ErrorText(
        error=(
            "The current lode was driven at {stored_dimension} dimensions, but this model "
            "yields {current_dimension}-wide nuggets — they don't sit on the same vein."
        ),
        hint=(
            "Re-mine it with `lode mine --from-scratch`, or switch your embedding config "
            "back to the model/dimension that dug it."
        ),
    ),
    "tokenizer_mismatch": ErrorText(
        error=(
            "The current lode was mined with {stored_tokenizer}, but now uses {current_tokenizer}"
            " — the lexical map no longer matches the terrain."
        ),
        hint=(
            "Re-mine it with `lode mine --from-scratch`, or switch your lexical config "
            "back to the tokenizer that mined it."
        ),
    ),
    "schema_version": ErrorText(
        error=("The lode's map belongs to an older era (schema {stored_version}), and today's tools cannot read it."),
        hint="Run `lode mine --from-scratch` to rebuild the index with the current schema.",
    ),
    "no_index": ErrorText(
        error="This workspace ({index_path}) has no lode yet.", hint="Run `lode mine` to mine it first."
    ),
    "invalid_digest": ErrorText(
        error="This digest cannot identify an ore: {digest!r}.",
        hint="Make sure the digest is valid.",
    ),
    "not_found": ErrorText(
        error="This lode contains no ore with digest {digest!r}.",
        hint="Make sure both the digest and the workspace are correct.",
    ),
    "ambiguous": ErrorText(
        error="There are {count} ores matching {digest!r}.", hint="Use a longer prefix to identify a single ore."
    ),
    "invalid_query": ErrorText(
        error="The lode cannot guess what you seek.", hint="Provide a query to start prospecting."
    ),
    "model_mismatch": ErrorText(
        error="The lode was shaped with a different model (indexed: {stored_model_id!r}).",
        hint="Run `lode mine --from-scratch` to reshape it, or switch back to that model.",
    ),
    "config_invalid": ErrorText(error="The lode cannot use this configuration: {detail}."),
    "extension_load": ErrorText(
        error=(
            "This Python cannot load SQLite extensions (python {python}, sqlite {sqlite_version}, "
            "SQLITE_OMIT_LOAD_EXTENSION={omit_load_extension}), so lode cannot load its native index extension."
        ),
        hint=(
            "Use a Python built with --enable-loadable-sqlite-extensions "
            "(e.g. `uv python install`), then run the command again."
        ),
    ),
}


def error_text(code: str, /, **fields: object) -> ErrorText | None:
    """Render the user-facing text for a store error code, or ``None``.

    Unknown codes have no template; callers fall back to the exception's own
    diagnostic message.
    """
    template = _TEMPLATES.get(code)
    if template is None:
        return None
    return ErrorText(template.error.format(**fields), template.hint.format(**fields))


def require_error_text(code: str, /, **fields: object) -> ErrorText:
    """Render the user-facing text for a code; a missing template is a bug.

    For fixed-shape CLI exits the code is written next to the call, so an
    unknown code means the table and the call site drifted apart — fail
    loudly instead of rendering nothing.
    """
    template = _TEMPLATES.get(code)
    if template is None:
        raise KeyError(f"no message template for code {code!r}")
    return ErrorText(template.error.format(**fields), template.hint.format(**fields))
