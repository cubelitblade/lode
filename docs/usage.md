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

Successful commands exit `0`; execution errors exit `1`, and Clap argument
errors exit `2` (`mine` uses `3` for a completed run with per-file failures).
For JSON execution errors, stdout contains one `{code,message,hint}` object and
stderr remains quiet unless logging was explicitly enabled. `assay why` uses
stable codes including `invalid_query`, `invalid_digest`, `invalid_top_k`,
`no_index`, `not_found`, `ambiguous`, `config_error`, `embedding_error`,
`dimension_error`, `index_open_error`, `index_read_error`, `scan_error`, and
`search_error`.

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

## `lode assay why`

Explain how one indexed chunk received its hybrid retrieval score. This
command supports all four Rust output views:

```bash
lode assay why DIGEST QUERY
lode --view extended assay why DIGEST QUERY
lode --view table assay why DIGEST QUERY
lode --view json assay why DIGEST QUERY [--top-k <number>]
lode --view json analyze why DIGEST QUERY [--top-k <number>]
```

`compact` gives the query, representative target, final score and rank, one
line per retrieval source, and the fusion method. `extended` adds all target
paths, source pool sizes and scores, normalization and fusion parameters, and
the core fusion explanation. It does not print the indexed chunk text.

`table` writes exactly two TSV records, one for `semantic` and one for
`lexical`, after this fixed header:

```text
QUERY TOP_K DIGEST PATH HEADING PAGE SEQ STATE COMBINED RANK IN_RESULTS SOURCE STATUS POOL_SIZE RAW_SCORE PREPARED_SCORE POOL_RANK FACTOR_METRIC FACTOR_VALUE NORM NORM_PARAMS FUSION FUSION_PARAMS
```

The displayed header uses spaces for readability; the actual fields are
tab-separated. Missing values are empty fields, booleans are `true` or
`false`, and parameters are compact JSON objects. Query, heading, and path
fields use the normal TSV escaping rules.

Rust ranks semantic cosine scores descending and original SQLite BM25 scores
ascending; equal scores use chunk rowid order. Lexical normalization preserves
the BM25 direction: min-max produces values in [-1, 0], and softmax produces
negative probabilities. Linear fusion uses the opposite of the prepared
lexical score times its weight, so a stronger lexical match contributes more.
The extended formula labels this term as `(-lexical) × weight`. RRF skips
normalization and ranks original lexical BM25 ascending. Raw BM25 and FTS phrase
contributions keep their original SQLite signs. This scoring correction needs
no index rebuild; it can change candidate membership and result scores.

The digest may be a full `blake3:<hex>` value, a bare hexadecimal value, or a
printed prefix. The command resolves exactly one indexed chunk, refreshes its
file freshness markers, and then scores the old indexed snapshot. A stale
result therefore reports `has_stale: true` but does not extract the changed
file or update its FTS rows or document vector; the query embedding may still
contact the configured embedding service.

The successful JSON object contains `query`, `top_k`, target provenance,
`has_stale`, per-source status and scores, the configured `norm` and `fusion`,
matched-source `ranking_factors`, and the final `combined`, `rank`, and
`in_results` values. `rank` is the target's rank across the complete candidate
pool, so it can exceed `top_k`; a target with no positive score has a null rank
and is not an execution error. A matched source's pool rank follows the same positional
tie-break used by RRF fusion. Human and table views warn on stderr when the
target is stale; JSON keeps stderr quiet. The explanation covers source scores,
normalization, and fusion. To include FTS phrase matches and signed contributions
to the raw BM25 score, explicitly request `--fts-details`:

```bash
lode --view json assay why DIGEST QUERY --fts-details
lode --view json analyze why DIGEST QUERY --fts-details --top-k 10
lode --view extended assay why DIGEST QUERY --fts-details
lode --view extended analyze why DIGEST QUERY --fts-details --top-k 10
```

This option supports JSON and extended views. Compact and table views reject
it with `unsupported_view` before opening the index, refreshing freshness
markers, or calling the embedding service. Without the option, all four views
and the existing JSON fields remain unchanged.

Extended output appends an independent FTS details section after the existing
fusion explanation. It shows the actual MATCH expression, query strategy and
physical tokenizer, full MATCH status, candidate-pool membership, raw BM25 and
corpus statistics. Each phrase retains its index, query tokens, hit count,
document frequency, IDF, weighted frequency, and signed BM25 contribution;
each occurrence shows its column, token offset, document tokens, and original
UTF-8 byte range and surface, or an explicit reason why mapping is unavailable.
Multi-token phrases stay together, repeated phrases remain separate, and
zero-contribution branches remain visible. Missing document tokens are shown
as `null` with an explicit `unknown document token` label.

The section explains that phrase contributions concern raw BM25. A matched
target outside the lexical candidate pool is explicitly marked as having a
BM25 score that did not participate in fusion. Disabled, empty, and nonmatching
queries explain why no statistics are displayed. Numeric values use the usual
four-decimal display rounding and scientific notation for small nonzero values;
these displays are not intended for exact summation. MATCH, token, and surface control
characters are rendered visibly.

The additional `lexical_details` object contains:

- `status`: `disabled`, `empty_query`, `not_matched`, or `matched`.
- `in_candidate_pool` and the actual `match_expression`.
- `tokenizer: {strategy, tokenize}`, distinguishing the query strategy from
  the physical tokenizer (`jieba` uses physical `simple`).
- `bm25`, `corpus_rows`, `document_length`, and `average_length`.
- `phrases`, each with `index`, `query_tokens`, `document_frequency`,
  `idf`, `weighted_frequency`, `bm25_contribution`, and `hits`.
  Hits contain `column`, `token_offset`, `document_tokens`,
  `mapping_status`, `mapping_reason`, `start_byte`, `end_byte`, and `surface`.

A phrase is an FTS5 scoring unit containing one or more tokenizer tokens; it
does not necessarily correspond to a natural-language word. Repeated phrases
and zero-contribution OR branches are retained. Contributions use SQLite's
signed BM25 convention and sum to `bm25` within
`1e-12 + 1e-12 * abs(bm25)`. They do not attribute normalized or fused
scores, including RRF. `token_offset` counts indexed token positions;
unavailable document tokens (such as prefix matches) remain JSON null.

Mapped hits have `mapping_status: "mapped"` and a null `mapping_reason`.
The zero-based `start_byte` is inclusive and `end_byte` exclusive: slicing the
indexed chunk's UTF-8 text at those boundaries gives exactly `surface`.
These are chunk-text offsets, including for PDF/DOCX, and do not locate bytes
in the original file. Stale results refer to the older indexed text.
Multi-token phrases cover their complete range; colocated pinyin variants
share a range and trigram ranges can overlap. Prefix surfaces come from the
actual document range even when their normalized document tokens are unknown.

Mapping uses the verified physical tokenizer and its DOCUMENT token stream in
the same read snapshot. Unknown table definitions or tokenizer parameters,
tokenization errors, invalid streams, or inconsistent positions withhold the
range instead of guessing. Such hits retain their score and token diagnostics,
with `mapping_status: "unavailable"`, a reason code, and null `start_byte`,
`end_byte`, and `surface`. Reason codes are `tokenizer_configuration`,
`tokenization_failed`, `invalid_token_stream`, `token_count_mismatch`,
`unsupported_column`, `hit_outside_document`, and `document_token_mismatch`.
The native tokenizer currently cannot map text containing an embedded NUL;
this produces `tokenization_failed`. Mapping unavailability is a successful
diagnostic result; a failure to obtain scoring diagnostics still returns the
execution error described below.

Full MATCH status is separate from candidate-pool membership. A matched target
outside the lexical pool gets diagnostics, while `sources.lexical.raw_score`
remains null and its existing fusion factors and rank remain unchanged. For
`disabled`, `empty_query`, and `not_matched`, statistics are null and
`phrases` is empty. Disabled queries have a null expression; empty lexical
queries have an empty expression and skip MATCH.

Candidates, the target, and diagnostics share one SQLite read snapshot after
freshness detection. The option makes no additional query-embedding request.
Explicit diagnostic failures return `fts_explanation_error` with exit status
1. JSON emits a single error object on stdout with quiet stderr; extended
emits the error on stderr with empty stdout. Neither returns a partial result.

Production-interface timing boundaries, reproduction commands and native-platform
validation status are documented in [FTS explanation performance](fts-explanation-performance.md).
The detail option executes on demand; its corpus phrase statistics can add cost
that grows with query length and document frequency.

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
