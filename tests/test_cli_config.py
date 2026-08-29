"""End-to-end CLI tests for the ``config`` command and config-loading failures.

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual typer app.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from lode import config
from lode.cli import app
from lode.cli.render import RenderOptions
from tests.conftest import runner


def _config_path(scope: str) -> Path:
    from lode.config import user_config_path, workspace_config_path

    return user_config_path() if scope == "user" else workspace_config_path()


def test_config_show_uses_configured_no_color(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config output also honours the configured output.no_color."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text("[app.output]\nno_color = true\n")

    captured: RenderOptions | None = None

    def fake_render(
        content: str,
        *,
        options: RenderOptions | None = None,
        console: Console | None = None,
    ) -> None:
        nonlocal captured
        captured = options

    monkeypatch.setattr("lode.cli.commands.config.render_config_show", fake_render)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured.no_color is True


def test_config_show_prints_effective_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[embedding]\nmodel = "m"\n')
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "[embedding]" in result.output
    assert 'model = "m"' in result.output


def test_config_show_subcommand_same_as_bare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    bare = runner.invoke(app, ["config"])
    show = runner.invoke(app, ["config", "show"])
    assert bare.exit_code == 0, bare.output
    assert show.exit_code == 0, show.output
    assert bare.output == show.output


def test_config_set_workspace_creates_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.model", "BAAI/bge-small-zh-v1.5"])
    assert result.exit_code == 0, result.output
    assert ".lode/config.toml" in result.output
    path = _config_path("workspace")
    assert path.is_file()
    assert 'model = "BAAI/bge-small-zh-v1.5"' in path.read_text()


def test_config_set_types_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.batch_size", "8"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["config", "set", "embedding.l2_normalize", "false"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["config", "set", "app.ignore.sources", ".gitignore, docs"])
    assert result.exit_code == 0, result.output

    data = config.read_toml(_config_path("workspace"))
    assert data["embedding"]["batch_size"] == 8
    assert data["embedding"]["l2_normalize"] is False
    assert data["app"]["ignore"]["sources"] == [".gitignore", "docs"]


def test_config_set_writes_existing_project_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lode.toml").write_text('[embedding]\nmodel = "old"\n')
    result = runner.invoke(app, ["config", "set", "embedding.model", "new"])
    assert result.exit_code == 0, result.output
    # Prefers the existing higher-precedence project file (lode.toml), not .lode/config.toml.
    assert "lode.toml" in result.output
    assert '"new"' in (tmp_path / "lode.toml").read_text()


def test_config_set_user_scope_writes_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["config", "set", "embedding.openai_compatible.endpoint", "http://x", "--scope", "user"]
    )
    assert result.exit_code == 0, result.output
    user_path = _config_path("user")
    assert user_path.is_file()
    assert 'endpoint = "http://x"' in user_path.read_text()


def test_config_set_unknown_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.unknown", "x"])
    assert result.exit_code != 0


def test_config_set_bad_type_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.batch_size", "abc"])
    assert result.exit_code != 0


def test_config_get_reads_merged_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[embedding]\nmodel = "m"\n')
    result = runner.invoke(app, ["config", "get", "embedding.model"])
    assert result.exit_code == 0, result.output
    assert 'embedding.model = "m"' in result.output


def test_config_get_scope_reads_layer_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Only user scope sets the key.
    runner.invoke(app, ["config", "set", "embedding.model", "user-model", "--scope", "user"])
    # Workspace layer has it unset -> fails.
    result = runner.invoke(app, ["config", "get", "embedding.model", "--scope", "workspace"])
    assert result.exit_code != 0
    # User layer returns the explicit value.
    result = runner.invoke(app, ["config", "get", "embedding.model", "--scope", "user"])
    assert result.exit_code == 0, result.output
    assert "user-model" in result.output


def test_config_unset_removes_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["config", "set", "embedding.model", "m"])
    result = runner.invoke(app, ["config", "unset", "embedding.model"])
    assert result.exit_code == 0, result.output
    data = config.read_toml(_config_path("workspace"))
    assert "model" not in data.get("embedding", {})
    assert '"model"' not in (tmp_path / ".lode" / "config.toml").read_text()


def test_config_unset_missing_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "unset", "embedding.model"])
    assert result.exit_code != 0


def test_config_path_shows_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0, result.output
    assert result.output.strip().endswith(".lode/config.toml")


# -- config loading failures reach the user as friendly exits ------------------
# catch_exceptions=False: an unhandled load_settings exception would propagate
# into the test and fail it; a handled failure exits cleanly with code 1.


def test_malformed_config_toml_fails_friendly(tmp_path: Path) -> None:
    """A syntactically broken config file exits cleanly, not with a traceback."""
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text("[embedding\nmodel = oops")

    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey"], catch_exceptions=False)

    assert result.exit_code == 1


def test_invalid_config_value_fails_friendly(tmp_path: Path) -> None:
    """A config value that fails validation exits cleanly, not with a traceback."""
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[embedding]\nbatch_size = "lots"\n')

    result = runner.invoke(app, ["--workspace", str(tmp_path), "survey"], catch_exceptions=False)

    assert result.exit_code == 1


def test_missing_explicit_config_path_fails_friendly(tmp_path: Path) -> None:
    """An explicit --config path that does not exist exits cleanly."""
    result = runner.invoke(
        app,
        ["--workspace", str(tmp_path), "survey", "--config", str(tmp_path / "nope.toml")],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
