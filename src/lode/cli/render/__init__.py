"""Rendering layer for lode CLI output.

Subpackage owns everything that turns command results into text: the style
core (``core.py``), shared output primitives (``output.py``), and per-command
renderers (``survey.py``, ``mine.py``, ``prospect.py``, ``dig.py``,
``config.py``).

The public surface is re-exported here so callers can use
``from lode.cli.render import render_survey, RenderOptions`` without reaching
into the internals.
"""

from __future__ import annotations

from lode.cli.render.assay import render_assay
from lode.cli.render.config import (
    render_config_message,
    render_config_path,
    render_config_set,
    render_config_show,
    render_config_unset,
    render_config_value,
)
from lode.cli.render.core import (
    ACCESSIBLE_DARK_INTENT_COLORS,
    ACCESSIBLE_LIGHT_INTENT_COLORS,
    ANSI_INTENT_COLORS,
    MARKERS,
    STATUS_INTENT,
    Border,
    Intent,
    RenderOptions,
    Status,
    render_options_from_preset,
)
from lode.cli.render.dig import render_dig
from lode.cli.render.mine import render_mine
from lode.cli.render.prospect import render_prospect
from lode.cli.render.survey import render_survey

__all__ = [
    "ACCESSIBLE_DARK_INTENT_COLORS",
    "ACCESSIBLE_LIGHT_INTENT_COLORS",
    "ANSI_INTENT_COLORS",
    "MARKERS",
    "STATUS_INTENT",
    "Border",
    "Intent",
    "RenderOptions",
    "Status",
    "render_assay",
    "render_config_message",
    "render_config_path",
    "render_config_set",
    "render_config_show",
    "render_config_unset",
    "render_config_value",
    "render_dig",
    "render_mine",
    "render_options_from_preset",
    "render_prospect",
    "render_survey",
]
