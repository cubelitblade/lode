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
    """Data state/outcome of an indexed file, as reported by survey/mine.

    ``FAILED`` is a per-run outcome for a file that could not be (re)indexed; it
    is reported by ``mine`` and shares the marker/intent system with the
    file-state values above.
    """

    NEW = "new"
    CHANGED = "changed"
    MISSING = "missing"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    FAILED = "failed"


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


# rich.box variant per Border. ``Border.NONE -> None`` signals "no frame": the
# renderer must then emit content without a Panel, because rich's Panel requires
# a ``box.Box`` and cannot take ``None``. Extending the enum adds an entry here.
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
    Status.FAILED: "×",  # noqa: RUF001 — intentional multiplication-sign glyph for errors
}

# Data state → presentation intent.
STATUS_INTENT: Mapping[Status, Intent] = {
    Status.NEW: Intent.SUCCESS,
    Status.CHANGED: Intent.WARNING,
    Status.MISSING: Intent.ERROR,
    Status.UNCHANGED: Intent.MUTED,
    Status.SKIPPED: Intent.MUTED,
    Status.FAILED: Intent.ERROR,
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
    Intent.SUCCESS: "spring_green1",
    Intent.ERROR: "dark_orange",
    Intent.WARNING: "gold1",
    Intent.INFO: "deep_sky_blue1",
    Intent.MUTED: "grey50",
}

# plain: every intent → empty style (no colour, no ANSI). Border is unaffected.
PLAIN_INTENT_COLORS: Mapping[Intent, str] = {intent: "" for intent in Intent}


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Styling knobs for command output.

    Colour (``intent_colors``) and border (``border``) are independent axes: the
    ``rich``/``plain``/``accessible`` presets only change colour, while ``border``
    (``none``/``round``/``square``) is set separately. Defaults reproduce the
    current rich output (rounded border, dim border, default colours).
    """

    border: Border = Border.ROUND
    intent_colors: Mapping[Intent, str] = field(default_factory=lambda: DEFAULT_INTENT_COLORS)
    # Border colour. Kept as a value (not a bool) so it can become configurable
    # later; ``dim`` keeps the frame from competing with the content colours.
    border_style: str = "dim"

    @property
    def box(self) -> box.Box | None:
        """The rich ``box`` variant backing ``border`` (``None`` = no frame)."""
        return _BORDER_BOX[self.border]


# Presets only control COLOUR; border is a separate axis (``RenderOptions.border``)
# left at the default. So ``plain`` is uncoloured but still bordered.
_PRESETS: Mapping[str, RenderOptions] = {
    "vivid": RenderOptions(),
    "plain": RenderOptions(intent_colors=PLAIN_INTENT_COLORS),
    "accessible": RenderOptions(intent_colors=ACCESSIBLE_INTENT_COLORS),
}


def render_options_from_preset(name: str) -> RenderOptions:
    """Resolve a palette name (``vivid``/``plain``/``accessible``) to options."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown render preset: {name!r}") from None
