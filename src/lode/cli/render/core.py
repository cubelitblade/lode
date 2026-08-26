"""Render configuration for lode CLI output.

This module owns the *styling* knobs that command output depends on. It is
deliberately decoupled from the data (``Status``) so an accessible
(colour-blind friendly) palette and a no-colour mode can swap the palette and
border without touching any render logic.

Design notes
------------
* Symbols (``MARKERS``) are always emitted, independent of colour — the same
  shape ships in every mode so information never relies on hue alone.
* ``Status`` is a data state; ``Intent`` is a presentation intent. The two are
  linked by ``STATUS_INTENT``. Extra intent roles (e.g. ``ERROR``/``WARNING``,
  used by non-survey commands later) slot into ``Intent`` + the colour tables,
  not into ``Status``.
* ``no_color`` (config or ``--no-color``) is applied at the ``Console`` layer
  (``Console(no_color=...)``), not by swapping in an empty palette: colour is
  off, but the layout stays structured.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from rich import box

# Change-list entries are plain paths except renames, which carry ``(old, new)``.
type Entry = str | tuple[str, str]


def entry_label(entry: Entry) -> str:
    """Display label for a change-list entry; renames render as ``old -> new``."""
    return f"{entry[0]} -> {entry[1]}" if isinstance(entry, tuple) else entry


class Status(StrEnum):
    """Data state/outcome of an indexed file, as reported by survey/mine.

    ``FAILED`` is a per-run outcome for a file that could not be (re)indexed; it
    is reported by ``mine`` and shares the marker/intent system with the
    file-state values above.
    """

    NEW = "new"
    CHANGED = "changed"
    MISSING = "missing"
    RENAMED = "renamed"
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
    # A rename does not touch content, so it carries no change glyph; the
    # ``old -> new`` pair itself is the signal. Placeholder kept intentionally
    # blank for now.
    Status.RENAMED: " ",
    Status.UNCHANGED: "=",
    Status.SKIPPED: "○",
    Status.FAILED: "×",  # noqa: RUF001 — intentional multiplication-sign glyph for errors
}

# Data state → presentation intent.
STATUS_INTENT: Mapping[Status, Intent] = {
    Status.NEW: Intent.SUCCESS,
    Status.CHANGED: Intent.WARNING,
    Status.MISSING: Intent.ERROR,
    Status.RENAMED: Intent.WARNING,
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


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Styling knobs for command output.

    Colour (``intent_colors``), border (``border``), and the colour on/off
    switch (``no_color``) are independent axes. ``intent_colors`` selects the
    palette (``vivid``/``accessible``); ``no_color`` (``True``/``False``/``None``)
    is applied at the ``Console`` layer — ``None`` defers to Rich's own
    ``NO_COLOR`` detection. Defaults reproduce the current rich output (rounded
    border, dim border, default colours).
    """

    border: Border = Border.ROUND
    intent_colors: Mapping[Intent, str] = field(default_factory=lambda: DEFAULT_INTENT_COLORS)
    # Border colour. Kept as a value (not a bool) so it can become configurable
    # later; ``dim`` keeps the frame from competing with the content colours.
    border_style: str = "dim"
    # Colour on/off switch, applied at the Console layer (not by swapping in an
    # empty palette). ``None`` = unset, defer to Rich's NO_COLOR detection.
    no_color: bool | None = None

    @property
    def box(self) -> box.Box | None:
        """The rich ``box`` variant backing ``border`` (``None`` = no frame)."""
        return _BORDER_BOX[self.border]


# Presets only control COLOUR; border (``RenderOptions.border``) and the
# no-colour switch (``RenderOptions.no_color``) are separate axes.
_PRESETS: Mapping[str, RenderOptions] = {
    "vivid": RenderOptions(),
    "accessible": RenderOptions(intent_colors=ACCESSIBLE_INTENT_COLORS),
}


def render_options_from_preset(name: str) -> RenderOptions:
    """Resolve a palette name (``vivid``/``accessible``) to options."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown render preset: {name!r}") from None
