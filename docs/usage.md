# Usage

## lode survey

Detect workspace changes and report stale files.

```bash
lode survey [OPTIONS] [workspace]
```

Aliases: `lode status`

This command only compares files and does not require an embedding endpoint.

### Options

- `[workspace]`
    Workspace to inspect.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `vivid|accessible`
    Color palette for this run.

- `--no-color`
   Disable color for this run.

- `--json`
   Emit JSON output.

### Output

Displays workspace status summary and affected files. Prefixes indicate file changes:
`+` new, `~` changed, `-` missing. A file moved without content changes is
reported as a rename (`old -> new`).

## lode mine

Embeds and indexes new or changed files. An embedding endpoint must be configured.

```bash
lode mine [OPTIONS] [workspace]
```

Aliases: `lode index`

Use `--from-scratch` when changing the embedding model or when the index
schema is incompatible.

### Options

- `[workspace]`
    - `<path>`
    Workspace to index.

- `--from-scratch`
  Discard the existing index and create a new one from scratch.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `vivid|accessible`
    Color palette for this run.

- `--no-color`
  Disable color for this run.

- `--json`
  Emit JSON output.

### Output

Displays indexing summary and affected paths. Prefixes indicate changes:
`+` added, `~` updated, `-` removed. A file moved without content changes is
re-pointed at zero embedding cost and reported as a rename (`old -> new`).

## lode prospect

Search the index and show results with source information.

```bash
lode prospect QUERY [OPTIONS] [workspace]
```

Aliases: `lode search`

Performs hybrid retrieval: semantic similarity (vector) + BM25 lexical
matching, combined by the configured `[norm]` / `[fusion]` plan (default:
min-max normalization + linear weighted fusion).

### Options

- `QUERY`
    - `<query>`
    Query to search for.

- `[workspace]`
    - `<path>`
    Workspace to search.

- `--top-k`
    - `<int>`
    Maximum number of results to return.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `vivid|accessible`
    Color palette for this run.

- `--no-color`
  Disable color for this run.

- `--json`
  Emit JSON output.

### Output

Displays ranked results with score, source location,
optional stale marker, and chunk digest, followed by a preview snippet.

## lode dig

Fetch a chunk's full text by its digest (content address).

```bash
lode dig DIGEST [OPTIONS] [workspace]
```

Aliases: `lode get`

The digest may be the full `blake3:<hex>`, a bare hex, or the short prefix
printed by `prospect` (optionally with a leading `#`). Ambiguous prefixes
return matching candidates.

### Options

- `DIGEST`
    - `<digest>`
    Chunk digest or prefix.

- `[workspace]`
    - `<path>`
    Workspace containing the index.

- `--radius`
    - `<int>`
    Number of adjacent chunks to include around the target chunk.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `vivid|accessible`
    Color palette for this run.

- `--no-color`
  Disable color for this run.

- `--json`
  Emit JSON output.

### Output

Displays the target chunk and its surrounding context window.

## lode assay

Explain why a chunk scored as it did for a query.

```bash
lode assay QUERY DIGEST [OPTIONS] [workspace]
```

Aliases: `lode analyze`

Reuses the same hybrid scoring as `prospect` (semantic + BM25, combined by
the configured `[norm]` / `[fusion]` plan) and breaks down the chunk's score
into its per-source raw and prepared values, the fusion, and its rank. A
chunk that did not make the results is explained as ranked outside `top_k`
or as having a zero combined score.

The digest may be the full `blake3:<hex>`, a bare hex, or the short prefix
printed by `prospect` (optionally with a leading `#`). Ambiguous prefixes
return matching candidates.

### Options

- `QUERY`
    - `<query>`
    Query to explain the score for.

- `DIGEST`
    - `<digest>`
    Chunk digest or prefix to explain.

- `[workspace]`
    - `<path>`
    Workspace to search.

- `--top-k`
    - `<int>`
    Maximum number of results to return.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `vivid|accessible`
    Color palette for this run.

- `--no-color`
  Disable color for this run.

- `--json`
  Emit JSON output.

### Output

Displays the query, the chunk's source location, per-source raw and prepared
scores with their normalization, the fusion, the combined score, and the
chunk's rank (or why it is not in the results).

## lode config

Inspect and modify configuration.

```bash
lode config [SUBCOMMAND] [OPTIONS]
```

A bare `lode config` defaults to `show`.

### Subcommands

- `show`
  Show the merged effective configuration (TOML).

- `get`
    - `<key>`
    Read a config value (merged effective value by default).

- `set`
    - `<key> <value>`
    Set a config value.

- `unset`
    - `<key>`
    Unset a config value from the target scope.

- `path`
  Show the target config file path.

### Options

- `--scope`
    - `user|workspace`
    Config scope.

- `--palette`
    - `vivid|accessible`
    Color palette for this run.

- `--no-color`
   Disable color for this run.
