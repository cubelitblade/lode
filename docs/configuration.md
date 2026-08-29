# Configuration

This document describes lode's configuration options.

## Loading

Lode loads configuration from the following sources, in order of priority:

1. CLI arguments
2. Environment variables
3. Workspace configuration files
4. User configuration file

All available configuration sources are loaded. When the same option is defined in multiple sources,
the value from the higher-priority source overrides the value from the lower-priority source.

### CLI arguments

Some commands provide CLI arguments as convenient shortcuts for configuring specific options. Values provided through CLI arguments apply only to the current invocation and override values from other configuration sources.

Not every configuration option is available as a CLI argument. See the documentation for each command for the options it supports.

For example:

```sh
lode survey --no-color
```

This command temporarily sets `app.output.no_color` to `true`.

### Environment variables

Lode-native environment variables can be used to configure any supported configuration option. They are derived from configuration paths by replacing each `.` with `__` and adding the `LODE_` prefix.

For example:

`embedding.model` → `LODE_EMBEDDING__MODEL`

Lode also supports the standard `NO_COLOR` environment variable. When it is set to a non-empty value, colored output is disabled.

For example:

```sh
NO_COLOR=1 lode survey
```

### Workspace configuration files

Lode loads all of the following workspace configuration files when they exist.
They are listed from highest to lowest priority:

1. `./.lode/config.toml`
2. `./lode.toml`
3. `./.lode.toml`

When the same option is defined in multiple workspace configuration files,
the value from the higher-priority file overrides the value from the lower-priority file.

### User configuration file

The user configuration file provides configuration shared across all workspaces.
Workspace configuration, environment variables, and CLI arguments can override its values.

* **Linux:** `~/.config/lode/config.toml`
* **macOS:** `~/Library/Application Support/lode/config.toml`
* **Windows:** `%APPDATA%\lode\config.toml`

## Options

### `version`

The schema version of the configuration file.

A config file may declare `version` at the top level. If omitted, lode assumes the current version, allowing existing config files to continue loading without modification.

If the declared version is not supported, lode fails with a clear error rather than silently misinterpreting configuration keys.

**Default:** `1`

### `chunking`

Controls how documents are split into chunks for indexing and embedding.

#### `size`

Maximum number of characters in each text chunk.

Larger chunks preserve more context but may reduce retrieval precision by introducing more unrelated content.

**Default:** `1024`

**Changing `size` requires re-mining the index.**

#### `overlap`

Number of overlapping characters between adjacent chunks.

Overlap helps preserve context when relevant information spans across chunk boundaries.

**Default:** `128`

**Changing `overlap` requires re-mining the index.**

### `fts`

Controls how the FTS5 lexical index is built and queried.

#### `strategy`

Selects the strategy used to build and query the lexical index.

Options:

- `unicode61`: SQLite's default tokenizer. Best suited for workspaces that mainly contain non-CJK documents.
- `trigram`: Uses 3-gram tokenization. Provides better substring matching for CJK text, especially for longer queries.
- `simple`: Uses the native extension with Han character and pinyin support. Default.
- `jieba`: Provides word-level matching using the `simple` index.

**Default:** `simple`

> [!NOTE]
> `simple` and `jieba` share the same underlying index tokenizer. Switching between them only changes query behavior and does not require re-mining the index.
>
> Other strategy changes require re-mining.

> [!TIP]
> Choose a strategy based on the language and query patterns of your workspace:
>
> - Use `unicode61` for workspaces that mainly contain non-CJK documents.
> - Use `trigram` when substring matching is important and queries are typically longer than 3 characters.
> - Use `simple` for mixed-language workspaces or when CJK and pinyin matching are needed.
> - Use `jieba` when word-level matching is preferred for Chinese text.
>
> For most typical workspaces, `simple` is recommended as a balanced default, providing good CJK support while retaining general-purpose full-text retrieval capabilities.

### `embedding`

Controls how embeddings are generated for semantic retrieval.

#### `provider`

Selects the backend used to generate embeddings.

Options:

- `openai_compatible`: An endpoint compatible with the OpenAI Embeddings API.
- `tei_native`: The native API of Hugging Face Text Embeddings Inference (TEI).
- `ollama`: The native Ollama API.

**Default:** `openai_compatible`

#### `model`

The name of the embedding model.

If omitted, lode attempts to discover the model information from the endpoint when supported.

> [!WARNING]
> On endpoints that expose multiple models, automatic discovery may select or probe a model that is not an embedding model.
>
> Set `model` explicitly when the endpoint hosts multiple models.

#### `model_dimension`

The native output dimension of the embedding model.

If omitted, lode attempts to discover it from the endpoint when supported.

The value should match the model's actual native output dimension.

**Changing the effective model dimension requires re-mining the index.**

#### `output_dimension`

The output dimension requested for each embedding.

When set, lode asks the provider to return vectors with this dimension instead of the model's full native output dimension.
This value should not exceed `model_dimension`.

Support depends on the selected provider, endpoint, and embedding model.
For example, some OpenAI-compatible APIs and TEI deployments may expose this through a dimensions parameter.

If omitted, lode uses the model's full native output dimension.

**Changing the effective output dimension requires re-mining the index.**

#### `l2_normalize`

Whether to apply L2 normalization to embedding vectors.

When enabled, cosine similarity can be computed using a dot product.

**Default:** `true`

**Usually recommended for retrieval tasks.**

#### `batch_size`

Number of texts sent to the embedding endpoint in a single request.

Larger values may improve throughput but increase memory usage and request size, and may exceed limits imposed by the embedding provider.

**Default:** `32`

#### `openai_compatible`

These options are used only when `embedding.provider` is set to `openai_compatible`.

##### `endpoint`

Base URL of the OpenAI-compatible API endpoint.

**Default:** `https://api.openai.com/v1`.

##### `key`

Optional API key for authenticated endpoints. When provided, it is sent in the `Authorization: Bearer <key>` header.

> [!IMPORTANT]
> Never store a key in a project-level configuration file that may be committed to version control.
>
> The user-level configuration file is not committed to version control and can be used to store keys.
> For better separation of secrets from configuration, prefer the environment variable `LODE_EMBEDDING__OPENAI_COMPATIBLE__KEY`.

##### `max_retries`

Maximum number of retries when an embedding request fails.

**Default:** `3`

##### `timeout`

Base timeout in seconds for each embedding request.

**Default:** `60.0`

#### `tei_native`

These options are used only when `embedding.provider` is set to `tei_native`.

##### `endpoint`

Base URL of the TEI native API endpoint.

**Default:** `http://localhost:8080`

##### `key`

Optional API key for authenticated endpoints. When provided, it is sent in the `Authorization: Bearer <key>` header.

> [!NOTE]
> TEI does not natively require an API key. This option is useful when TEI is deployed behind a proxy or gateway that requires authentication.

> [!IMPORTANT]
> Never store a key in a project-level configuration file that may be committed to version control.
>
> The user-level configuration file is not committed to version control and can be used to store keys.
> For better separation of secrets from configuration, prefer the environment variable `LODE_EMBEDDING__TEI_NATIVE__KEY`.

##### `max_retries`

Maximum number of retries when an embedding request fails.

**Default:** `3`

##### `timeout`

Base timeout in seconds for each embedding request.

**Default:** `60.0`

##### `truncate`

Whether to truncate inputs longer than the model's maximum length.

**Default:** `false`

##### `truncation_direction`

Which side of the input to truncate when `truncate` is set to `true`.

Options:

- `left`
- `right`

**Default:** `right`

#### `ollama`

These options are used only when `embedding.provider` is set to `ollama`.

##### `endpoint`

Base URL of the Ollama endpoint.

**Default:** `http://localhost:11434`

##### `key`

Optional API key for authenticated endpoints. When provided, it is sent in the `Authorization: Bearer <key>` header.

> [!IMPORTANT]
> Never store a key in a project-level configuration file that may be committed to version control.
>
> The user-level configuration file is not committed to version control and can be used to store keys.
> For better separation of secrets from configuration, prefer the environment variable `LODE_EMBEDDING__OLLAMA__KEY`.

##### `max_retries`

Maximum number of retries when an embedding request fails.

**Default:** `3`

##### `timeout`

Base timeout in seconds for each embedding request.

**Default:** `60.0`

##### `truncate`

Whether to truncate inputs longer than the model's maximum length.

**Default:** `true`

### `retrieval`

Controls options for retrieving results from the index.

#### `top_k`

Maximum number of candidates returned by `lode prospect`.
Larger values provide more candidates.

**Default:** `10`

> [!TIP]
> `lode prospect` returns previews rather than the full contents of each chunk, so increasing `retrieval.top_k` does not directly increase the amount of chunk content returned to the caller.
> Keep this value large enough to provide useful candidates for an agent or user to choose from. Use `lode dig` to retrieve the full contents of a selected chunk.

#### `norm`

Controls how scores from each retrieval source are normalized before fusion.
Normalization is skipped when `fusion.type` is set to `rrf`.

##### `type`

The score normalization method applied to each source before fusion.

Options:

- `minmax`: Normalizes each source's scores to `[0, 1]` using min-max scaling.
- `softmax`: Normalizes each source's scores using softmax.

**Default:** `softmax`

##### `softmax`

These options are used only when `norm.type` is set to `softmax`.

###### `temperature`

Controls the sharpness of the softmax distribution.

Higher values flatten the distribution, while lower values make it sharper.

**Default:** `1.0`

#### `fusion`

Controls how scores from different retrieval sources are combined.

##### `type`

The method used to combine scores from different retrieval sources.

Options:

- `linear`: Combines scores using a weighted sum.
- `rrf`: Combines results using reciprocal rank fusion.

**Default:** `linear`

##### `linear`

These options are used only when `fusion.type` is set to `linear`.

###### `semantic_factor`

Controls the importance of semantic similarity.

**Default:** `0.7`

###### `lexical_factor`

Controls the importance of lexical similarity.

**Default:** `0.3`

> [!NOTE]
> When `linear` is used as the fusion strategy, lode calculates a final score for each result using a weighted sum, then ranks the results by that score.
>
> $$
> S_{\text{final}}
> =
> S_{\text{semantic}} \times F_{\text{semantic}}
> +
> S_{\text{lexical}} \times F_{\text{lexical}}
> $$
>
> Where $S$ denotes a score and $F$ denotes a factor.
>
> Factors may be negative for advanced ranking strategies, allowing a retrieval source to reduce a result's final score.
>
> They must not both be zero.

##### `rrf`

These options are used only when `fusion.type` is set to `rrf`.

###### `k`

Controls how strongly rank differences affect the final score.

Higher values reduce the difference in contribution between higher- and lower-ranked results. Lower values give more weight to higher-ranked results.

**Default:** `60`

> [!NOTE]
> When `rrf` is used as the fusion strategy, lode calculates a final score based on ranks.
>
> $$
> S_{\text{final}}
> =
> \frac{1}{k+R_{\text{semantic}}}
> +
> \frac{1}{k+R_{\text{lexical}}}
> $$
>
> Where $S$ denotes a score and $R$ denotes a rank. $k$ controls how strongly rank differences affect the final score.

### `app`

Controls general application behavior.

#### `ignore`

Controls how lode matches and applies ignore rules when indexing a workspace.

##### `sources`

Additional files containing ignore patterns.

For example: `.gitignore`, `.dockerignore`.

`.lodeignore` is always supported and does not need to be listed.

#### `output`

Controls how lode formats and displays terminal output.

##### `no_color`

Disable colors in terminal output.

- `true`: Never use colors.
- `false`: Always use colors.
- unset: Follow the standard `NO_COLOR` environment variable.

**Default:** unset

##### `palette`

The color palette used for terminal output.

Options:

- `ansi`: Uses the terminal's standard ANSI color palette.
- `accessible_light`: A color-distinguishable palette optimized for light terminal backgrounds, based on Paul Tol’s vibrant color scheme.
- `accessible_dark`: A color-distinguishable palette optimized for dark terminal backgrounds, based on Okabe-Ito color scheme.

**Default:** `ansi`

> [!NOTE]
> Lode uses an intent-based color system for terminal output. Colors are assigned according to the role of a message in the user interaction flow rather than the meaning of the message content itself.
>
> Supported intents include:
>
> - `success`: Positive feedback indicating that an operation completed successfully or produced a desired result.
> - `warning`: A non-fatal issue that may require attention but does not prevent the operation from continuing.
> - `error`: A failure or problem that prevents an operation from completing successfully.
> - `info`: General information that helps users understand the current operation or state.
>
> For example, `success` does not necessarily mean that the message describes a business success state. It is used for positive feedback in the user interaction flow.
>
> The appearance of colors depends on terminal capabilities:
>
> - The `ansi` palette relies on colors defined by the terminal emulator. Different terminals and themes may render the same ANSI color differently.
> - The `accessible_light` and `accessible_dark` palettes use explicit RGB colors. They require true color (24-bit color) terminal support to preserve the intended appearance. On terminals without true color support, colors may be approximated and appear differently.
>
> The following table shows representative color values for each palette:
>
> | Intent | `ansi`                                                                       | `accessible_light`                                                                              | `accessible_dark`                                                                                         |
> | -------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
> | `success`       | ![ansi_success_green](https://img.shields.io/badge/ansi_green-00ff00)   | ![accessible_light_success_teal](https://img.shields.io/badge/teal-009988)     | ![accessible_dark_success_bluish_green](https://img.shields.io/badge/bluish_green-009e73) |
> | `warning`       | ![ansi_warning_yellow](https://img.shields.io/badge/ansi_yellow-ffff00) | ![accessible_light_warning_orange](https://img.shields.io/badge/orange-ee7733) | ![accessible_dark_warning_yellow](https://img.shields.io/badge/yellow-f0e442)             |
> | `error`       | ![ansi_error_red](https://img.shields.io/badge/ansi_red-ff0000)         | ![accessible_light_error_red](https://img.shields.io/badge/red-cc3311)         | ![accessible_dark_error_vermilion](https://img.shields.io/badge/vermilion-d55e00)         |
> | `info`       | ![ansi_info_cyan](https://img.shields.io/badge/ansi_cyan-00ffff)        | ![accessible_light_info_blue](https://img.shields.io/badge/blue-0077bb)        | ![accessible_dark_info_sky_blue](https://img.shields.io/badge/sky_blue-56b4e9)            |
