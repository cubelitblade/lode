# FTS5 phrase-level BM25 feasibility probe

Validated on 2026-10-02 against baseline commit `7b1130b`.

## Outcome

Phrase-level decomposition is feasible with Lode's bundled SQLite and all four
query strategies. The probe checks 98 scenarios and 669 matched rows. Every
phrase-contribution sum agrees with native `bm25()` within `1e-12 + 1e-12 *
abs(bm25)`. The largest observed absolute error is
`3.552713678800501e-15`. A further 83 target-row checks preserve the corpus row
count, phrase document frequencies and score when adding a `rowid` predicate.

This is a runnable experiment under
`lode-rs/crates/lode-core/examples/fts_bm25_probe/`, with no production API,
CLI, dependency or schema change. It uses synthetic in-memory indexes and
never opens the workspace index or requests embeddings.

## Reproduce

From `lode-rs/`:

```sh
cargo run --release --locked -p lode-core --example fts_bm25_probe > /tmp/lode-fts-bm25.json
```

The JSON includes every MATCH expression, query phrase's normalized tokens,
document frequency, IDF, weighted frequency, signed BM25 contribution, matched
columns, token offsets and document tokens when available. It also includes
compiler/SQLite versions, tolerances, case counts and timing summaries. These
fields belong to the experiment and are not a proposed public JSON contract.

The parent process supplies a temporary `LODE_CACHE_DIR` to its child and owns
cleanup after exit. A second release run with the parent's `LODE_CACHE_DIR`
set to `/dev/null` produced identical correctness evidence; timing samples
are expected to vary. Raw reports were saved locally under
`.ai/process/fts-bm25-eval/20261002/` and can be regenerated with the command above.

## Scoring and evidence

The auxiliary function reads FTS5's actual query context: `xInst` supplies
phrase occurrences, `xQueryPhrase` supplies corpus document frequencies, and
column-size APIs supply document and average lengths. It mirrors bundled
SQLite's BM25 constants (`k1=1.2`, `b=0.75`), nonpositive-IDF replacement
(`1e-6`), column weights and negative sign. Each contribution is signed so
that its sum equals SQL `bm25()`. Query-wide statistics are cached with
`xSetAuxdata` and released by FTS5 at statement completion.

The correctness fixture uses the same external-content FTS layout as Lode;
extra two-column fixtures exercise weights and column filters. Queries come
from the existing `match_query` and real native helpers and are checked
against `explain_query`. Additional explicit MATCH fixtures cover AND, OR,
NOT, NEAR, phrases, prefixes, repeated phrases, unmatching branches, no
matches, IDF clamping and zero weights. Native fixtures also cover Chinese
phrases, pinyin matching and colocated variants. Query execution leaves the
fixture's total-change counter unchanged.

Observed semantics:

- `unicode61`: query `知识` produces one token/phrase in this fixture.
- `simple`: `知识` produces `"知" AND "识"`, with two scoring phrases.
- `jieba`: `知识` produces `"知识"`, one scoring phrase containing the two
  underlying character tokens. There is no built-in per-character share of
  this phrase's score.
- `trigram`: fewer than three characters produces an empty expression through
  Lode's builder. Longer input produces overlapping three-character phrases.
- Repeated query phrases remain separate scoring units; deduplicating them
  would change native BM25 semantics. The probe independently checks their
  equal contributions and expected occurrence frequencies.
- Native `"知" OR "zhi"` can count distinct scoring phrases at the same token
  position. `zhi` also matches the alternate reading of `识` in this fixture.
  Matching normalized tokens therefore cannot be treated as original words.
- `xInstToken` in bundled SQLite 3.46.0 returns no document token for a prefix
  match. The probe records `null` while retaining phrase identity, position,
  frequency and exact score. Original-text span mapping and prefix surface
  recovery remain separate work.

One existing query-construction defect was reproduced: trigram input
`alpha " beta` generates an improperly escaped MATCH expression, and SQLite
returns `unterminated string`. The baseline and diagnostic queries reject it
identically. `unicode61`, `simple` and `jieba` accept that input in this fixture.
The probe records the rejection; it does not repair query construction.

## Cost

Environment: stable `rustc 1.98.1 (48a229cea 2026-09-01)`, SQLite 3.46.0,
Linux x86_64 / WSL2 kernel 6.18.40.1-microsoft-standard-WSL2,
Intel Core i7-12700H, 6 virtual CPUs. All values below come from the release
build, after the other quality checks finished.

Each measurement targets row 1 in an external-content in-memory index of
1,000 or 10,000 synthetic chunks. Phrases are deliberately common to exercise
corpus-wide document-frequency scans. Statements, index pages and dictionary
are warm; each execution starts new FTS query statistics. There is one warmup
per variant and 21 samples with alternating order. The table shows medians in
microseconds. The baseline selects native BM25; the diagnostic selects BM25
plus the JSON-producing auxiliary function. SQL preparation, indexing,
dictionary initialization, helper construction, JSON parsing and output are
excluded. These are target-query timings, not end-to-end `assay why` timings.

| Strategy | Corpus rows | BM25 us | With probe us | Additional us | Ratio |
|---|---:|---:|---:|---:|---:|
| unicode61 | 1,000 | 30.840 | 61.047 | 30.207 | 1.98 |
| unicode61 | 10,000 | 290.446 | 596.753 | 306.307 | 2.05 |
| trigram | 1,000 | 131.407 | 261.026 | 129.619 | 1.99 |
| trigram | 10,000 | 1172.059 | 2307.978 | 1135.919 | 1.97 |
| simple | 1,000 | 167.277 | 274.867 | 107.590 | 1.64 |
| simple | 10,000 | 1563.318 | 2507.509 | 944.191 | 1.60 |
| jieba | 1,000 | 191.522 | 311.341 | 119.819 | 1.63 |
| jieba | 10,000 | 1837.412 | 3091.309 | 1253.897 | 1.68 |

The extra document-frequency pass is visible even for a single target. Its
statistics cache is separate from native BM25's cache, so selecting both
functions duplicates some work. At 10,000 rows, measured extra time is
0.306–1.254 ms. Different strategies produce different expressions; their
absolute times do not isolate tokenizer speed. Synthetic corpus results do
not establish performance on larger or disk-backed user indexes.

## Decision boundary

A next implementation can expose phrase matches and signed raw-BM25
contributions on demand. Preserve phrase identity, repeated query units,
colocated variants and the distinction between normalized tokens and source
text. Keep the current retrieval scoring authoritative.

This experiment does not define contributions for natural-language words,
normalized source scores or final hybrid scores. In particular, a BM25 phrase
share is not an RRF share. It also does not diagnose partially matched rows
that fail the overall MATCH expression, parse the boolean condition tree, or
recover source-character spans.

Before CLI integration, decide the original-query/source-text mapping,
prefix recovery, lexical pool cutoff versus actual MATCH semantics, output
contract and acceptable on-demand cost. Windows and macOS builds/execution
remain unverified by this local experiment.

## Validation

Passed: debug and release probe runs, invalid-parent-cache isolation run,
`cargo fmt --all --check`, strict workspace Clippy with all targets/features,
`cargo test --workspace`, `cargo deny --locked check`, `cargo audit`, and
whitespace checks. Deny reports existing duplicate-dependency warnings;
audit reports only the already allowed unmaintained `ttf-parser` advisory.

References: [FTS5 auxiliary APIs](https://www.sqlite.org/fts5.html#custom_auxiliary_functions)
and [SQLite 3.46 BM25 implementation](https://sqlite.org/src/file?name=ext/fts5/fts5_aux.c&ci=version-3.46.0).
