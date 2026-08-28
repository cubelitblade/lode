"""Unit tests for `lode.cli.render.config`.

Config output is presentation, not a stable contract, so these tests assert the
verbatim text is emitted (no wrapping/re-colouring) rather than exact style
details.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from lode.cli.render.config import (
    render_config_message,
    render_config_path,
    render_config_set,
    render_config_show,
    render_config_unset,
    render_config_value,
)


def _render(run: Callable[[Console], None]) -> str:
    """Render a config output to plain text via a recording console."""
    console = Console(record=True, force_terminal=False)
    run(console)
    return console.export_text()


def test_render_config_show_emits_toml() -> None:
    text = _render(lambda console: render_config_show('[embedding]\nmodel = "m"\n', console=console))
    assert "[embedding]" in text
    assert 'model = "m"' in text


def test_render_config_value_emits_key_value() -> None:
    text = _render(
        lambda console: render_config_value("embedding.openai_compatible.endpoint", "https://x", console=console)
    )
    assert 'embedding.openai_compatible.endpoint = "https://x"' in text


def test_render_config_value_formats_bool_lowercase() -> None:
    text = _render(lambda console: render_config_value("embedding.l2_normalize", False, console=console))
    assert "embedding.l2_normalize = false" in text


def test_render_config_set_confirmation() -> None:
    text = _render(lambda console: render_config_set("embedding.model", "m", Path("/tmp/lode.toml"), console=console))
    assert 'set embedding.model = "m" in /tmp/lode.toml' in text


def test_render_config_unset_confirmation() -> None:
    text = _render(lambda console: render_config_unset("embedding.model", Path("/tmp/lode.toml"), console=console))
    assert "unset embedding.model in /tmp/lode.toml" in text


def test_render_config_path() -> None:
    text = _render(lambda console: render_config_path(Path("/tmp/lode.toml"), console=console))
    assert "/tmp/lode.toml" in text


def test_render_config_message_error() -> None:
    text = _render(lambda console: render_config_message("Dry hole: unknown config key 'x'.", console=console))
    assert "Dry hole: unknown config key 'x'." in text
