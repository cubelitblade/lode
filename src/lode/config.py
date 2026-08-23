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

import json
import types
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin

import platformdirs
import tomlkit
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
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
    re-chunked automatically — run ``lode mine --from-scratch`` after changing
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


# -- config file editing helpers (used by the `lode config` CLI) ------------


def workspace_config_path() -> Path:
    """Path of the project-level config file to write.

    Prefers the highest-precedence existing project file; if none exists,
    falls back to ``.lode/config.toml`` (the runtime data dir).
    """
    for path in reversed(PROJECT_CONFIG_PATHS):
        if path.is_file():
            return path
    return PROJECT_CONFIG_PATHS[-1]


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file into a plain dict (empty if missing)."""
    if not path.is_file():
        return {}
    return tomlkit.loads(path.read_text(encoding="utf-8")).unwrap()


def write_toml(path: Path, data: dict[str, Any]) -> None:
    """Write a plain dict as TOML, preserving comments/format via tomlkit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(data), encoding="utf-8")


def get_nested(data: dict[str, Any], key: str) -> Any:
    """Look up a dotted key in a nested dict; raise KeyError if absent."""
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = cast(dict[str, Any], node)[part]
    return node


def set_nested(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted key in a nested dict, creating intermediate sections."""
    parts = key.split(".")
    node: dict[str, Any] = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = cast(dict[str, Any], child)
    node[parts[-1]] = value


def unset_nested(data: dict[str, Any], key: str) -> bool:
    """Remove a dotted key from a nested dict; return True if it existed."""
    parts = key.split(".")
    node: dict[str, Any] = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            return False
        node = cast(dict[str, Any], child)
    if parts[-1] not in node:
        return False
    del node[parts[-1]]
    return True


# Sections a user may read/write via `lode config`; excludes internal fields
# such as `config_files`.
CONFIG_SECTIONS = ("embedding", "retrieval", "chunking", "ignore")

# Boolean tokens accepted by `lode config set <key> <bool-value>`.
_BOOL_TRUE = frozenset({"true", "1", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"false", "0", "no", "n", "off"})


def _unwrap_optional(annotation: Any) -> Any:
    """Reduce `Optional[X]` (either spelling) to `X`.

    pydantic stores field annotations as written in the model: a
    ``str | None`` field keeps a ``types.UnionType`` origin while an explicit
    ``Optional[str]`` keeps ``typing.Union``, so both must be handled.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _field_for_key(model: type[BaseModel], key: str) -> FieldInfo | None:
    """Resolve a dotted config key to its leaf pydantic ``FieldInfo``."""
    parts = key.split(".")
    current: type[BaseModel] = model
    field: FieldInfo | None = None
    for index, part in enumerate(parts):
        field = current.model_fields.get(part)
        if field is None:
            return None
        if index < len(parts) - 1:
            nested = _unwrap_optional(field.annotation)
            if isinstance(nested, type) and issubclass(nested, BaseModel):
                current = nested
            else:
                return None
    return field


def validate_key(key: str) -> None:
    """Raise ``KeyError`` if ``key`` is not a settable leaf config key."""
    parts = key.split(".")
    if not parts or not all(parts):
        raise KeyError(key)
    if parts[0] not in CONFIG_SECTIONS:
        raise KeyError(key)
    field = _field_for_key(Settings, key)
    if field is None:
        raise KeyError(key)
    annotation = _unwrap_optional(field.annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        # A whole section (`embedding`, `embedding.api`) is not a settable leaf.
        raise KeyError(key)


def _type_name(annotation: Any) -> str:
    """Human-readable name for a config field type (for error messages)."""
    if get_origin(annotation) is list:
        return "list"
    name = getattr(annotation, "__name__", None)
    return name if isinstance(name, str) else repr(annotation)


def _parse_bool(key: str, value: str) -> bool:
    token = value.strip().lower()
    if token in _BOOL_TRUE:
        return True
    if token in _BOOL_FALSE:
        return False
    raise ValueError(f"{value!r} is not a valid value for {key} (expected bool)")


def _parse_list(key: str, value: str) -> list[str]:
    """Parse a CLI list: comma-separated or a JSON array."""
    token = value.strip()
    if token.startswith("["):
        try:
            parsed = json.loads(token)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{value!r} is not a valid value for {key} (expected a JSON array)") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{value!r} is not a valid value for {key} (expected a list of strings)")
        result: list[str] = []
        for item in cast(list[Any], parsed):
            if not isinstance(item, str):
                raise ValueError(f"{value!r} is not a valid value for {key} (expected a list of strings)")
            result.append(item)
        return result
    return [item.strip() for item in token.split(",") if item.strip()]


def parse_value(key: str, value: str) -> Any:
    """Coerce a CLI string to the config field's Python type."""
    field = _field_for_key(Settings, key)
    if field is None:
        raise KeyError(key)
    annotation = _unwrap_optional(field.annotation)
    if annotation is bool:
        return _parse_bool(key, value)
    if get_origin(annotation) is list:
        return _parse_list(key, value)
    try:
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not a valid value for {key} (expected {_type_name(annotation)})") from exc
    return value


def effective_config(settings: Settings) -> dict[str, Any]:
    """Return the merged effective config as a nested dict (for ``show``)."""
    return settings.model_dump()


def _without_none(node: object) -> Any:
    """Recursively drop ``None`` values so a plain dict is TOML-serialisable."""
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in cast(dict[str, Any], node).items():
            if value is None:
                continue
            result[key] = _without_none(value)
        return result
    if isinstance(node, list):
        return [_without_none(value) for value in cast(list[Any], node)]
    return node


def toml_dumps(data: dict[str, Any]) -> str:
    """Serialize a nested dict to TOML text, omitting unset (``None``) values."""
    return tomlkit.dumps(_without_none(data))


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
