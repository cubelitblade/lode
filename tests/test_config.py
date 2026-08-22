"""Tests for layered settings loading (TOML file + env vars).

Hermetic: config files are written to tmp_path and LODE_* env vars are
pinned via monkeypatch, so no test depends on the host environment or the
current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lode import config
from lode.embeddings.openai_compat import OpenAICompatibleEmbedder


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture: run for every test, not referenced directly
    """Remove any LODE_* vars and isolate the user config dir."""
    for key in list(os.environ):
        if key.startswith(config.ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    # Redirect the user-level config so host configs don't leak into tests.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _write_toml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_defaults_without_file_or_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = config.load_settings()
    assert settings.embedding == config.EmbeddingConfig()
    assert settings.retrieval == config.RetrievalConfig()
    assert settings.chunking == config.ChunkingConfig()
    assert settings.ignore == config.IgnoreConfig()


def test_default_path_reads_partial_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[embedding]
model = "BAAI/bge-small-zh-v1.5"
batch_size = 8
""",
    )
    settings = config.load_settings()
    assert settings.embedding.model == "BAAI/bge-small-zh-v1.5"
    assert settings.embedding.batch_size == 8
    # Fields not present in the file keep their defaults.
    assert settings.embedding.api.endpoint == "http://localhost:8080"
    # Sections absent from the file fall back to their defaults entirely.
    assert settings.retrieval == config.RetrievalConfig()
    assert settings.chunking == config.ChunkingConfig()


def test_chunking_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[chunking]
size = 2048
overlap = 256
""",
    )
    settings = config.load_settings()
    assert settings.chunking.size == 2048
    assert settings.chunking.overlap == 256


def test_chunking_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[chunking]
size = 2048
overlap = 256
""",
    )
    monkeypatch.setenv("LODE_CHUNKING__SIZE", "4096")
    settings = config.load_settings()
    assert settings.chunking.size == 4096
    # Env vars only touch the fields they name.
    assert settings.chunking.overlap == 256


def test_explicit_path(tmp_path: Path) -> None:
    toml = tmp_path / "custom.toml"
    _write_toml(toml, '[embedding.api]\nendpoint = "http://127.0.0.1:11434"\n')
    settings = config.load_settings(toml)
    assert settings.embedding.api.endpoint == "http://127.0.0.1:11434"


def test_explicit_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        config.load_settings(tmp_path / "missing.toml")


def test_nested_api_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[embedding.api]
type = "openai_compatible"
endpoint = "http://127.0.0.1:11434"
max_retries = 5
timeout = 30.0
""",
    )
    settings = config.load_settings()
    assert settings.embedding.api.type == "openai_compatible"
    assert settings.embedding.api.endpoint == "http://127.0.0.1:11434"
    assert settings.embedding.api.max_retries == 5
    assert settings.embedding.api.timeout == 30.0


def test_nested_api_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[embedding.api]
endpoint = "http://127.0.0.1:11434"
""",
    )
    monkeypatch.setenv("LODE_EMBEDDING__API__ENDPOINT", "http://0.0.0.0:9999")
    settings = config.load_settings()
    assert settings.embedding.api.endpoint == "http://0.0.0.0:9999"


def test_user_config_loads_as_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    user_path = config.user_config_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text('[embedding]\nmodel = "user-model"\n')
    settings = config.load_settings()
    assert settings.embedding.model == "user-model"


def test_project_config_overrides_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    user_path = config.user_config_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text('[embedding]\nmodel = "user-model"\n')
    (tmp_path / "lode.toml").write_text('[embedding]\nmodel = "project-model"\n')
    settings = config.load_settings()
    assert settings.embedding.model == "project-model"


def test_project_configs_merge_later_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode.toml").write_text('[embedding]\nmodel = "dot-model"\nbatch_size = 1\n')
    (tmp_path / "lode.toml").write_text('[embedding]\nmodel = "root-model"\n')
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[embedding]\nmodel = "local-model"\n')
    settings = config.load_settings()
    # .lode/config.toml wins (latest), but batch_size from .lode.toml survives (merge).
    assert settings.embedding.model == "local-model"
    assert settings.embedding.batch_size == 1


def test_explicit_path_skips_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lode.toml").write_text('[embedding]\nmodel = "discovered"\n')
    custom = tmp_path / "custom.toml"
    custom.write_text('[embedding]\nmodel = "explicit"\n')
    settings = config.load_settings(custom)
    assert settings.embedding.model == "explicit"


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[embedding]
model = "from-file"
batch_size = 8
""",
    )
    monkeypatch.setenv("LODE_EMBEDDING__MODEL", "from-env")
    settings = config.load_settings()
    assert settings.embedding.model == "from-env"
    # Env vars only touch the fields they name.
    assert settings.embedding.batch_size == 8


def test_env_without_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LODE_RETRIEVAL__TOP_K", "25")
    settings = config.load_settings()
    assert settings.retrieval.top_k == 25
    assert settings.retrieval.semantic_factor == config.DEFAULT_SEMANTIC_FACTOR


def test_explicit_kwargs_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LODE_EMBEDDING__MODEL", "from-env")
    settings = config.Settings(embedding=config.EmbeddingConfig(model="from-kwarg"))
    assert settings.embedding.model == "from-kwarg"


def test_build_embedder_from_settings() -> None:
    settings = config.Settings(
        embedding=config.EmbeddingConfig(api=config.EmbeddingApiConfig(endpoint="http://localhost:9999"))
    )
    embedder = config.build_embedder(settings.embedding)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.base_url == "http://localhost:9999"


# -- helpers for the `lode config` CLI -----------------------------------------


def test_validate_key_accepts_leaf_keys() -> None:
    config.validate_key("embedding.model")
    config.validate_key("embedding.api.endpoint")
    config.validate_key("retrieval.semantic_factor")
    config.validate_key("chunking.size")
    config.validate_key("ignore.sources")


def test_validate_key_rejects_sections_and_unknown() -> None:
    # Whole sections are not settable leaves.
    for key in ("embedding", "embedding.api", "retrieval", "chunking", "ignore"):
        with pytest.raises(KeyError):
            config.validate_key(key)
    # Unknown/invalid keys.
    for key in ("config_files", "unknown", "embedding.unknown", ""):
        with pytest.raises(KeyError):
            config.validate_key(key)


def test_parse_value_types() -> None:
    assert config.parse_value("embedding.model", "BAAI/bge-small-zh-v1.5") == "BAAI/bge-small-zh-v1.5"
    assert config.parse_value("embedding.batch_size", "8") == 8
    assert config.parse_value("embedding.api.timeout", "30.0") == 30.0
    assert config.parse_value("embedding.l2_normalize", "false") is False
    assert config.parse_value("embedding.l2_normalize", "1") is True
    assert config.parse_value("ignore.sources", ".gitignore, docs") == [".gitignore", "docs"]
    assert config.parse_value("ignore.sources", '["a", "b"]') == ["a", "b"]


def test_parse_value_rejects_bad_types() -> None:
    with pytest.raises(ValueError, match=r"embedding\.batch_size"):
        config.parse_value("embedding.batch_size", "abc")
    with pytest.raises(ValueError, match=r"embedding\.l2_normalize"):
        config.parse_value("embedding.l2_normalize", "maybe")
    with pytest.raises(ValueError, match=r"ignore\.sources"):
        config.parse_value("ignore.sources", "[1, 2]")
    with pytest.raises(KeyError):
        config.parse_value("unknown.key", "x")


def test_effective_config_drops_internal_field() -> None:
    settings = config.Settings(
        embedding=config.EmbeddingConfig(model="m", api=config.EmbeddingApiConfig(endpoint="http://x")),
        chunking=config.ChunkingConfig(size=2048),
    )
    data = config.effective_config(settings)
    # The internal `config_files` field must not leak into the serialized config.
    assert "config_files" not in data
    assert data["embedding"]["model"] == "m"
    assert data["embedding"]["api"]["endpoint"] == "http://x"
    assert data["chunking"]["size"] == 2048
