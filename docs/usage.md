# Usage

## Global options

Options that apply to every command and must appear before the command name:

- `--workspace`, `-C`
    - `<path>`
    Workspace to operate on. Defaults to the current directory.

## lode survey

Detect workspace changes and report stale files.

```bash
lode survey [OPTIONS]
```

Aliases: `lode status`

This command only compares files and does not require an embedding endpoint.

### Options

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `ansi|accessible_light|accessible_dark`
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
lode mine [OPTIONS]
```

Aliases: `lode index`

Use `--from-scratch` when changing the embedding model or when the index
schema is incompatible.

### Options

- `--from-scratch`
  Discard the existing index and create a new one from scratch.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `ansi|accessible_light|accessible_dark`
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
lode prospect QUERY [OPTIONS]
```

Aliases: `lode search`

Performs hybrid retrieval: semantic similarity (vector) + BM25 lexical
matching, combined by the configured `[norm]` / `[fusion]` plan (default:
min-max normalization + linear weighted fusion).

### Options

- `QUERY`
    - `<query>`
    Query to search for.

- `--top-k`
    - `<int>`
    Maximum number of results to return.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `ansi|accessible_light|accessible_dark`
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
lode dig DIGEST [OPTIONS]
```

Aliases: `lode get`

The digest may be the full `blake3:<hex>`, a bare hex, or the short prefix
printed by `prospect` (optionally with a leading `#`). Ambiguous prefixes
return matching candidates.

### Options

- `DIGEST`
    - `<digest>`
    Chunk digest or prefix.

- `--radius`
    - `<int>`
    Number of adjacent chunks to include around the target chunk.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `ansi|accessible_light|accessible_dark`
    Color palette for this run.

- `--no-color`
  Disable color for this run.

- `--json`
  Emit JSON output.

### Output

Displays the target chunk and its surrounding context window.

## lode assay

Explain how one indexed chunk is processed and scored. Two views:

- `lode assay why DIGEST QUERY` — why the chunk scored as it did for a query.
- `lode assay how DIGEST` — how the configured lexical analyzer splits the
  chunk's text at index time.

```bash
lode assay why DIGEST QUERY [OPTIONS]
lode assay how DIGEST [OPTIONS]
```

Aliases: `lode analyze why|how`

`why` reuses the same hybrid scoring as `prospect` (semantic + BM25, combined
by the configured `[norm]` / `[fusion]` plan) and breaks down the chunk's
score into its per-source raw and prepared values, the fusion, and its rank.
A chunk that did not make the results is explained as ranked outside `top_k`
or as having a zero combined score.

The digest may be the full `blake3:<hex>`, a bare hex, or the short prefix
printed by `prospect` (optionally with a leading `#`). Ambiguous prefixes
return matching candidates.

### Options (`why`)

- `DIGEST`
    - `<digest>`
    Chunk digest or prefix to explain.

- `QUERY`
    - `<query>`
    Query to explain the score for.

- `--top-k`
    - `<int>`
    Maximum number of results to return.

- `--config`
    - `<path>`
    Path to a configuration file.

- `--palette`
    - `ansi|accessible_light|accessible_dark`
    Color palette for this run.

- `--no-color`
  Disable color for this run.

- `--json`
  Emit JSON output.

### Output (`why`)

Displays the query, a brief pipeline overview (per-source metric, candidate
pool size and within-pool rank, the norm→fusion stage, and the final rank),
the chunk's source location, per-source raw and prepared scores with their
normalization, the fusion, the combined score, and the chunk's rank (or why
it is not in the results).

### Options (`how`)

- `DIGEST`
    - `<digest>`
    Chunk digest or prefix to inspect.

- `--config`, `--palette`, `--no-color`, `--json`
    Same as `why`.

### Output (`how`)

Displays the chunk's provenance, the lexical layer as two roles —
**Lexical analyzer** (the configured strategy, which decides query-side
analysis) and **Storage tokenizer** (the FTS5 `tokenize=` clause, the index
storage form) — a preview of the processed text, and the distinct indexed
terms in their original order. The term line is truncated by rendered width
(visual units), not by term count; `--json` always carries the complete raw
token stream plus the structured `terms` list. The stream comes from SQLite
itself — an in-memory FTS5 table with the same tokenizer read back through
`fts5vocab` — so it always matches what indexing produced, including case
folding and pinyin expansion for the native strategies.

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
    - `ansi|accessible_light|accessible_dark`
    Color palette for this run.

- `--no-color`
   Disable color for this run.
