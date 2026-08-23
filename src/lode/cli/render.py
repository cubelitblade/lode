"""Render configuration for lode CLI output.

This module owns the *styling* knobs that command output depends on. It is
deliberately decoupled from the data (``Status``) so a future plain-text or
accessible (colour-blind friendly) mode can swap the palette and border without
touching any render logic.

Design notes
------------
* Symbols (``MARKERS``) are always emitted, independent of colour — the same
  shape ships in every mode so information never relies on hue alone.
* ``Status`` is a data state; ``Intent`` is a presentation intent. The two are
  linked by ``STATUS_INTENT``. Extra intent roles (e.g. ``ERROR``/``WARNING``,
  used by non-survey commands later) slot into ``Intent`` + the colour tables,
  not into ``Status``.
* ``plain`` mode uses ``PLAIN_INTENT_COLORS`` (every intent → empty style) plus
  ``Border.NONE``: no colour, no border, but the layout stays structured.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from rich import box


class Status(StrEnum):
    """Data state of an indexed file, as reported by survey/mine."""

    NEW = "new"
    CHANGED = "changed"
    MISSING = "missing"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"


class Intent(StrEnum):
    """Presentation intent/semantics of a piece of output.

    Mirrors the narrative used by help panels and the JSON envelope
    (``success``/``error``), so colours can be assigned by role rather than by
    data state. This is what lets us add ``Error``/``Warning`` styling later
    without touching ``Status``.
    """

    SUCCESS = "success"
    ERROR = "error"
    INFO = "info"
    WARNING = "warning"
    MUTED = "muted"


class Border(StrEnum):
    """Border style for rendered panels/tables.

    ``rich.box`` variant chosen by ``RenderOptions.box``. Keep it an enum so a
    CLI option (``--style``/``--border``) and a config value map cleanly onto a
    finite set instead of a free-form string.
    """

    NONE = "none"
    ROUND = "round"
    SQUARE = "square"


# rich.box variant per Border. ``None`` means no border (rich uses ``box=None``),
# not a sentinel ``box.Box`` — rich has no ``box.NONE`` constant. Extending the
# enum means adding an entry here.
_BORDER_BOX: Mapping[Border, box.Box | None] = {
    Border.NONE: None,
    Border.ROUND: box.ROUNDED,
    Border.SQUARE: box.SQUARE,
}

# Symbols are always emitted, regardless of colour, so the output stays
# decipherable in plain/accessible modes. ``skipped`` uses ``○`` (an
# "empty/no-op" glyph) so the rendering code never special-cases "no marker".
MARKERS: Mapping[Status, str] = {
    Status.NEW: "+",
    Status.CHANGED: "~",
    Status.MISSING: "-",
    Status.UNCHANGED: "=",
    Status.SKIPPED: "○",
}

# Data state → presentation intent.
STATUS_INTENT: Mapping[Status, Intent] = {
    Status.NEW: Intent.SUCCESS,
    Status.CHANGED: Intent.WARNING,
    Status.MISSING: Intent.ERROR,
    Status.UNCHANGED: Intent.MUTED,
    Status.SKIPPED: Intent.MUTED,
}

# Default (rich) colours: actionable signals stand out, passive ones recede.
DEFAULT_INTENT_COLORS: Mapping[Intent, str] = {
    Intent.SUCCESS: "green",
    Intent.ERROR: "red",
    Intent.WARNING: "yellow",
    Intent.INFO: "cyan",
    Intent.MUTED: "dim",
}

# accessible (colour-blind friendly): distinct hues with strong luminance
# separation, paired with the always-on symbols so hue is never the only cue.
ACCESSIBLE_INTENT_COLORS: Mapping[Intent, str] = {
    Intent.SUCCESS: "blue",
    Intent.ERROR: "magenta",
    Intent.WARNING: "cyan",
    Intent.INFO: "cyan",
    Intent.MUTED: "dim",
}

# plain: every intent → empty style (no colour, no ANSI), no border.
PLAIN_INTENT_COLORS: Mapping[Intent, str] = {intent: "" for intent in Intent}


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Styling knobs for command output.

    Defaults reproduce the current rich output: rounded border, dim border,
    default intent colours. Callers replace ``border``/``intent_colors`` (via a
    preset or directly) to switch into plain/accessible modes.
    """

    border: Border = Border.ROUND
    intent_colors: Mapping[Intent, str] = field(default_factory=lambda: DEFAULT_INTENT_COLORS)
    # Border colour. Kept as a value (not a bool) so it can become configurable
    # later; ``dim`` keeps the frame from competing with the content colours.
    border_style: str = "dim"

    @property
    def box(self) -> box.Box | None:
        """The rich ``box`` variant backing ``border`` (``None`` = no border)."""
        return _BORDER_BOX[self.border]


_PRESETS: Mapping[str, RenderOptions] = {
    "rich": RenderOptions(),
    "plain": RenderOptions(border=Border.NONE, intent_colors=PLAIN_INTENT_COLORS),
    "accessible": RenderOptions(intent_colors=ACCESSIBLE_INTENT_COLORS),
}


def render_options_from_preset(name: str) -> RenderOptions:
    """Resolve a preset name (``rich``/``plain``/``accessible``) to options."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown render preset: {name!r}") from None
