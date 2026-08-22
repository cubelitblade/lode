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
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture: run for every test, not referenced directly
    """Remove any LODE_* vars so every test starts from a blank slate."""
    for key in list(os.environ):
        if key.startswith(config.ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)


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
    assert settings.embedding.base_url == "http://localhost:8080"
    # Sections absent from the file fall back to their defaults entirely.
    assert settings.retrieval == config.RetrievalConfig()
    assert settings.chunking == config.ChunkingConfig()


def test_chunking_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[chunking]
chunk_size = 2048
chunk_overlap = 256
""",
    )
    settings = config.load_settings()
    assert settings.chunking.chunk_size == 2048
    assert settings.chunking.chunk_overlap == 256


def test_chunking_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(
        tmp_path / ".lode" / "config.toml",
        """
[chunking]
chunk_size = 2048
chunk_overlap = 256
""",
    )
    monkeypatch.setenv("LODE_CHUNKING__CHUNK_SIZE", "4096")
    settings = config.load_settings()
    assert settings.chunking.chunk_size == 4096
    # Env vars only touch the fields they name.
    assert settings.chunking.chunk_overlap == 256


def test_explicit_path(tmp_path: Path) -> None:
    toml = tmp_path / "custom.toml"
    _write_toml(toml, '[embedding]\nbase_url = "http://127.0.0.1:11434"\n')
    settings = config.load_settings(toml)
    assert settings.embedding.base_url == "http://127.0.0.1:11434"


def test_explicit_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        config.load_settings(tmp_path / "missing.toml")


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
    assert settings.retrieval.dense_weight == config.DEFAULT_DENSE_WEIGHT


def test_explicit_kwargs_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LODE_EMBEDDING__MODEL", "from-env")
    settings = config.Settings(embedding=config.EmbeddingConfig(model="from-kwarg"))
    assert settings.embedding.model == "from-kwarg"


def test_build_embedder_from_settings() -> None:
    settings = config.Settings(embedding=config.EmbeddingConfig(base_url="http://localhost:9999"))
    embedder = config.build_embedder(settings.embedding)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.base_url == "http://localhost:9999"
