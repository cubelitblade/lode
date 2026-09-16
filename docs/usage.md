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

## Reserved commands

`lode assay` and `lode config` are recognised command names but are not
implemented in the Rust binary yet. They return an execution error until their
command handlers are added.
