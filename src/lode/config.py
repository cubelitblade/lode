"""Runtime configuration.

Config is the composition root: the rest of the code depends on interfaces
(Embedder, Splitter, ...) and this module decides which concrete
implementation is selected.

Settings are layered, lowest to highest precedence:

    defaults < TOML file < environment variables < explicit constructor kwargs

The TOML source is pluggable: `load_settings()` reads `.lode/config.toml`
by default, or an explicit path. Future file formats (YAML, JSON) can be added
as extra sources in `Settings.settings_customise_sources`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

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
# Default config file location, relative to the current working directory.
DEFAULT_CONFIG_PATH = Path(".lode") / "config.toml"

# Defaults for hybrid retrieval (dense + sparse); consumed in M2.
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_SPARSE_WEIGHT = 0.4
DEFAULT_TOP_K = 10


class EmbeddingConfig(BaseModel):
    """Settings for the embedding model backend."""

    provider: str = "openai_compat"
    base_url: str = "http://localhost:8080"
    model: str | None = None
    dimension: int | None = None
    batch_size: int = 32
    timeout: float = 60.0
    retries: int = 3
    normalize: bool = True


class RetrievalConfig(BaseModel):
    """Settings for hybrid retrieval. Reserved for M2 — not read yet."""

    dense_weight: float = DEFAULT_DENSE_WEIGHT
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT
    top_k: int = DEFAULT_TOP_K


class IgnoreConfig(BaseModel):
    """Settings for file-discovery ignore rules.

    Rather than listing patterns inline, config names ignore-like files to
    load (gitignore semantics). ``.lodeignore`` is always loaded when present
    at the workspace root; ``files`` adds extra sources such as ``.gitignore``.
    Reserved for M2 — not read yet.
    """

    files: list[str] = Field(default_factory=list)


def _init_kwargs(source: PydanticBaseSettingsSource) -> dict[str, Any]:
    """Return the init kwargs of a settings source.

    pydantic-settings does not type `init_kwargs` in its stubs, so accessing
    it is a known exemption; centralizing the cast keeps the rest of the
    module type-clean.
    """
    return cast(dict[str, Any], source.init_kwargs)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # pydantic-settings stub omits init_kwargs


class Settings(BaseSettings):
    """Layered application settings: TOML file + environment variables.

    Precedence, highest first: explicit constructor kwargs, environment
    variables (`LODE_` prefix, `__` section delimiter), the TOML file,
    then model defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
    )

    # Config file path passed by load_settings(); not a real setting, so it is
    # excluded from serialization and env-var lookup.
    config_file: Path | None = Field(default=None, exclude=True)

    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
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
        """Insert the TOML file between env vars and defaults.

        The file path is passed as a regular `config_file` init field and
        popped here, so callers stay type-safe while the file itself never
        becomes a settings attribute.
        """
        init_kwargs = _init_kwargs(init_settings)
        config_file = init_kwargs.pop("config_file", None)
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if config_file is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=Path(config_file)))
        return (*sources, dotenv_settings, file_secret_settings)


def load_settings(toml_path: str | Path | None = None) -> Settings:
    """Load settings from a TOML file (plus env vars), or bare defaults.

    With no argument, reads `.lode/config.toml` when present — a missing
    default file is not an error. An explicitly given path must exist.
    """
    if toml_path is None:
        if not DEFAULT_CONFIG_PATH.is_file():
            return Settings()
        return Settings(config_file=DEFAULT_CONFIG_PATH)
    path = Path(toml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return Settings(config_file=path)


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Construct the embedding implementation selected by config."""
    if cfg.provider == "openai_compat":
        return OpenAICompatibleEmbedder(
            base_url=cfg.base_url,
            model=cfg.model,
            dimension=cfg.dimension,
            batch_size=cfg.batch_size,
            timeout=cfg.timeout,
            retries=cfg.retries,
            normalize=cfg.normalize,
        )
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
