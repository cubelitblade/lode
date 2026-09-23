# Usage

The Rust `lode` binary indexes UTF-8, UTF-16, and Latin-1 text files,
Markdown, DOCX, and digital PDFs. DOCX extraction preserves paragraph order,
built-in heading chains, and table rows as pipe-delimited text. PDF extraction
preserves non-empty page numbers and outline headings for single-flow digital
documents.

PDF table reconstruction, OCR/scanned pages, general multi-column or complex
layout reading order, and general RTL PDFs are not supported. Image-only PDF
pages are accepted but produce no text. Corrupt or unauthenticated encrypted
DOCX/PDF files are reported as per-file mining failures and do not discard the
rest of a run.

## Global options

These options apply to every command and may appear before or after the
command name.

- `--workspace`, `-C <path>`
  Workspace to operate on. Defaults to the current directory.
- `--view <compact|extended|table|json>`
  Result view written to stdout. Defaults to `compact`.
- `--log-level <error|warn|info|debug>`
  Process log level written to stderr.
- `-v`, `-vv`
  Shorthand for `--log-level info` and `--log-level debug`.
- `--palette <ansi|accessible_tol|accessible_okabe_ito>`
  Colour palette for human-readable output.
- `--no-color`
  Disable colour for this invocation while retaining textual emphasis.
- `--no-progress`
  Disable the `mine` progress indicator.

`table` writes stable tab-separated records and `json` writes structured JSON.
Neither view adds terminal styling. Human views use terminal capability
detection and do not emit ANSI sequences when output is redirected or when
`TERM=dumb`.

## `lode survey`

Detect workspace changes and report files that are new, changed, removed, or
renamed.

```bash
lode survey
lode status
```

`status` is an alias for `survey`. The command compares files without calling
an embedding endpoint. With no index, supported files are classified as new.

## `lode mine`

Embed and index new or changed files.

```bash
lode mine [--from-scratch]
lode index [--from-scratch]
```

`index` is an alias for `mine`. `--from-scratch` archives the existing index
and rebuilds it using the current workspace settings. Progress is shown only
for human views when stderr is an interactive, non-dumb terminal. A run with
per-file failures returns status `3` and still reports successful and failed
entries in its result.

## `lode prospect`

Search the index with hybrid dense and lexical retrieval.

```bash
lode prospect QUERY [--top-k <number>]
lode search QUERY [--top-k <number>]
```

`search` is an alias for `prospect`. Results include their score, source
location, digest, and a text preview. `--top-k` limits the number of returned
results.

## `lode dig`

Fetch a stored chunk and optional neighbouring chunks by digest.

```bash
lode dig DIGEST [--radius <number>]
lode get DIGEST [--radius <number>]
```

`get` is an alias for `dig`. The digest may be a full `blake3:<hex>` value, a
bare hexadecimal value, or a printed short prefix. Ambiguous prefixes return
the matching candidates. `--radius` controls how many chunks are included on
each side of the target.

## `lode assay how`

Explain how one lexical query is constructed under the current workspace
configuration. This command does not require `.lode/index.db`, scan the
workspace, or contact an embedding service:

```bash
lode assay how '知识'
lode --view extended analyze how 'rock and ore'
lode --view table assay how '矿石探矿'
lode --view json assay how 'deadbeef'
```

The input is always query text, even when it looks like a digest. The result
shows the configured strategy, physical tokenizer, native helper when used,
and the actual lexical MATCH expression. The explanation describes lexical
query construction only; it does not report existing index hits or ranking.

`unicode61` quotes word fragments and joins them with `OR`; `trigram` builds
three-character windows joined with `OR`; `simple` builds native `AND`
expressions with its pinyin and prefix rules; `jieba` segments the query at
query time while using the same physical `simple` index tokenizer. A query
shorter than three characters can therefore produce an empty trigram
expression without being an execution error.

The JSON result contains `query`, `tokenizer`, `helper`, and
`match_expression`. The table view has the fixed columns `QUERY`, `STRATEGY`,
`TOKENIZER`, `HELPER`, and `MATCH`.

## `lode debug tokens`

Inspect how one indexed chunk was tokenized at index time. This is the
advanced index diagnostic that retains the chunk-oriented views:

```bash
lode debug tokens DIGEST
lode --view extended debug tokens DIGEST
lode --view table debug tokens DIGEST
lode --view json debug tokens DIGEST
```

The digest may be a full `blake3:<hex>` value, a bare hexadecimal value, or a
printed short prefix. The result contains the stored text, all source paths
and freshness states, the configured strategy and physical tokenizer, the
occurrence-preserving token stream, and the de-duplicated terms. Native
`simple` and `jieba` output colocated pinyin forms in `variants`.

This diagnostic scans the workspace to refresh stale markers, but it does not
update chunks, FTS rows, or embeddings; run `lode mine` to update the index.

## `lode config`

Inspect and edit configuration without opening an index or contacting an
embedding service:

```bash
lode config show
lode config get embedding.model
lode config get embedding.model --scope user
lode config path --scope workspace
lode config set fts.strategy unicode61 --scope workspace
lode config unset embedding.model --scope user
```

`set`, `unset`, and `path` default to workspace scope. Workspace edits use the
highest-precedence existing project file, or `.lode/config.toml` when none
exists. User edits use the platform user config path. `show` and `get` support
`compact`, `extended`, `table`, and `json` views; API keys are always redacted.
Rust applies L2 normalization unconditionally. The old
`embedding.l2_normalize` TOML key and its `LODE_` environment variable are
ignored.

The Rust quickstart should set the currently implemented FTS tokenizer
explicitly before mining:

```bash
lode config set fts.strategy unicode61 --scope workspace
lode survey
lode mine
lode prospect "your question"
```

`lode assay` prints a concise help message when no subcommand is provided.
