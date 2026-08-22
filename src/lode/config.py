"""Runtime configuration.

Config is the composition root: the rest of the code depends on interfaces
(Embedder, Splitter, ...) and this module decides which concrete
implementation is selected.

Settings are layered, lowest to highest precedence:

    defaults < user TOML < project TOMLs < environment variables < kwargs

`load_settings()` discovers a user-level config (via platformdirs) plus
project-level `.lode.toml`, `lode.toml`, and `.lode/config.toml`, merging them
so project-local files override the user file. An explicit path skips
discovery. Future file formats (YAML, JSON) can be added as extra sources in
`Settings.settings_customise_sources`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import platformdirs
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from lode.embeddings.base import Embedder
from lode.embeddings.openai_compat import OpenAICompatibleEmbedder

# Environment variables are read with this prefix, e.g. LODE_EMBEDDING__MODEL.
ENV_PREFIX = "LODE_"
# Nested config sections are addressed with a double underscore in env vars,
# e.g. LODE_EMBEDDING__MODEL (a single underscore would be ambiguous with
# the underscores already inside field names).
ENV_NESTED_DELIMITER = "__"
# Project-level config file candidates, in ascending precedence (later wins).
PROJECT_CONFIG_PATHS = (
    Path(".lode.toml"),
    Path("lode.toml"),
    Path(".lode") / "config.toml",
)

# Defaults for hybrid retrieval (semantic + lexical); consumed in M2.
DEFAULT_SEMANTIC_FACTOR = 0.7
DEFAULT_LEXICAL_FACTOR = 0.3
DEFAULT_TOP_K = 10

# Defaults for text chunking; consumed by the ingestion pipeline.
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128


class EmbeddingApiConfig(BaseModel):
    """Transport settings for the embedding HTTP backend."""

    type: str = "openai_compatible"
    key: str | None = None
    endpoint: str = "http://localhost:8080"
    max_retries: int = 3
    timeout: float = 60.0


class EmbeddingConfig(BaseModel):
    """Settings for the embedding model backend."""

    model: str | None = None
    dimension: int | None = None
    l2_normalize: bool = True
    batch_size: int = 32
    api: EmbeddingApiConfig = EmbeddingApiConfig()


class RetrievalConfig(BaseModel):
    """Settings for hybrid retrieval. Reserved for M2 — not read yet."""

    semantic_factor: float = DEFAULT_SEMANTIC_FACTOR
    lexical_factor: float = DEFAULT_LEXICAL_FACTOR
    top_k: int = DEFAULT_TOP_K


class ChunkingConfig(BaseModel):
    """Settings for text chunking.

    Changing these values changes how files are split into chunks, but the
    index is keyed by file digest, so already-indexed files are not
    re-chunked automatically — run ``lode mine --rebuild`` after changing
    them.
    """

    size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP


class IgnoreConfig(BaseModel):
    """Settings for file-discovery ignore rules.

    Rather than listing patterns inline, config names ignore-like files to
    load (gitignore semantics). ``.lodeignore`` is always loaded when present
    at the workspace root; ``files`` adds extra sources such as ``.gitignore``.
    Reserved for M2 — not read yet.
    """

    sources: list[str] = Field(default_factory=list)


def _init_kwargs(source: PydanticBaseSettingsSource) -> dict[str, Any]:
    """Return the init kwargs of a settings source.

    pydantic-settings does not type `init_kwargs` in its stubs, so accessing
    it is a known exemption; centralizing the cast keeps the rest of the
    module type-clean.
    """
    return cast(dict[str, Any], source.init_kwargs)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # pydantic-settings stub omits init_kwargs


def _default_config_files() -> list[Path]:
    return []


class Settings(BaseSettings):
    """Layered application settings: TOML files + environment variables.

    Precedence, highest first: explicit constructor kwargs, environment
    variables (`LODE_` prefix, `__` section delimiter), the TOML files,
    then model defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
    )

    # Config file paths passed by load_settings() (ascending precedence); not
    # a real setting, so it is excluded from serialization and env-var lookup.
    config_files: list[Path] = Field(default_factory=_default_config_files, exclude=True)

    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ignore: IgnoreConfig = IgnoreConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the TOML sources between env vars and defaults.

        Paths are passed as a regular `config_files` init field and popped
        here. The list is in ascending precedence; sources are appended in
        reverse so project-local files override the user file, while env and
        init kwargs still take precedence over all TOML sources.
        """
        init_kwargs = _init_kwargs(init_settings)
        config_files = [Path(p) for p in cast(list[Any], init_kwargs.pop("config_files", []))]
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        for path in reversed(config_files):
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=path))
        return (*sources, dotenv_settings, file_secret_settings)


def load_settings(toml_path: str | Path | None = None) -> Settings:
    """Load settings from TOML files (plus env vars), or bare defaults.

    With no argument: discover a user-level config (platformdirs) and
    project-level `.lode.toml`, `lode.toml`, `.lode/config.toml`, merging them
    so project-local files override the user file. Missing files are not an
    error. An explicitly given path must exist and skips discovery.
    """
    if toml_path is not None:
        path = Path(toml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return Settings(config_files=[path])
    return Settings(config_files=_discover_config_files())


def user_config_path() -> Path:
    """Cross-platform user-level config file path."""
    return Path(platformdirs.user_config_dir("lode")) / "config.toml"


def _discover_config_files() -> list[Path]:
    """Existing config files in ascending precedence (user, then project)."""
    candidates: list[Path] = [user_config_path(), *PROJECT_CONFIG_PATHS]
    return [path for path in candidates if path.is_file()]


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Construct the embedding implementation selected by config."""
    if cfg.api.type == "openai_compatible":
        return OpenAICompatibleEmbedder(
            base_url=cfg.api.endpoint,
            api_key=cfg.api.key,
            model=cfg.model,
            dimension=cfg.dimension,
            batch_size=cfg.batch_size,
            timeout=cfg.api.timeout,
            retries=cfg.api.max_retries,
            normalize=cfg.l2_normalize,
        )
    raise ValueError(f"Unknown embedding provider: {cfg.api.type!r}")
