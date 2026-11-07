# FTS explanation performance and validation

Measured on 2026-10-03 with the production FTS diagnostic and original-text
mapping functions. The historical feasibility probe remains in
[fts-bm25-evaluation.md](fts-bm25-evaluation.md); its auxiliary timing is a
different boundary and is not the current production performance result.

The recorded measurements predate the Rust BM25 ordering and contribution
correction. Rerun the commands below to measure the corrected candidate pools
and scoring behavior; the historical timings remain unchanged.

## Reproduce

From lode-rs, use the repository stable toolchain and an absolute output path
on the filesystem to be measured. The disk fixtures are created in a temporary
scratch subdirectory alongside that output and are removed after the run.

    LODE_FTS_PERFORMANCE_REPORT=/absolute/output/performance.json cargo test --release --locked -p lode-core --lib index::search::fts::performance::production_performance -- --exact --ignored --nocapture
    LODE_FTS_JSON_REPORT=/absolute/output/public-json.json cargo test --release --locked -p lode-cli --bin lode lexical_json::performance::public_json_performance -- --exact --ignored --nocapture

LODE_FTS_PERFORMANCE_SIZES can override the default 1000,10000,100000 sizes
for a smoke run. Reports include all observations, exact compiler and SQLite
versions, OS/architecture, queries, physical expression, phrase/hit counts,
target size and lexical-pool membership. These opt-in ignored tests assert
score/range consistency and embedding counts, never hardware-specific latency
thresholds. Their measurements are not part of the public CLI JSON contract.

The core runner supplies a temporary LODE_CACHE_DIR to a child process and
owns dictionary-cache cleanup, including when the caller has an unusable
cache setting. Public JSON measurements use unicode61 and need no dictionary.

## Measurement boundaries

- Baseline calls the existing production search::explain, with dense/lexical
  retrieval and fusion. Semantic queries use the in-process FakeEmbedder:
  no HTTP, a constant query vector, four-dimensional stored vectors, exactly
  one query embedding per complete pipeline.
- Diagnostics without mapping composes the same production candidate,
  explanation, Store.explain_fts, pool-validation and read-snapshot functions.
  It omits only hit mapping and the small display-metadata wrapper. It does
  not implement another BM25 formula or change candidate limits.
- Complete calls production explain_with_fts, including the snapshot,
  original-text mapper and typed metadata assembly.
- Mapping-only calls production Store.map_fts_hits on an already obtained
  diagnostic and target text. It includes schema/parameter verification,
  DOCUMENT tokenization, validation and range/surface allocation; cloning
  the diagnostic, corpus statistics and query embedding are outside its timer.
- Core diagnostic timings include private auxiliary JSON encoding/decoding.
  They exclude the public CLI adapter, outer why JSON fields, stdout,
  terminal rendering, process startup, fixture construction and Store opening.
- The JSON runner separately measures production lexical_json::details plus
  serde_json::to_vec for the complete lexical_details object. Retrieval,
  mapping, outer why fields, stdout and process startup are excluded.

Result checks and output-report serialization occur outside the timers.
Rotating pipeline order reduces ordering drift. Differences of independently
sampled medians are useful context and can be negative due to noise; the
mapping-only observations are the direct measurement of mapping work.
Complete minus diagnostics also includes display metadata, so it is not an
exact substitute for mapping-only timing.

## Corpus and cache conditions

The matrix uses four query strategies (unicode61, trigram, simple and jieba),
1k/10k/100k chunks, and memory and disk indexes. Each corpus contains a short
target and a target with 512 repeated text groups. Other rows contain common
terms plus unique filler text; needleabc occurs only in the two targets.
Queries cover frequent, rare and longer mixed-language input. Native alpha
queries exercise the real helper's prefix branches against alpha/alphabet;
the generated MATCH expression is recorded without rewriting other strategies.

Warm observations use the same connection after untimed oracle runs, with
nine samples per pipeline. Disk observations additionally use three first-query
samples on independently reopened connections; opening/schema/native
registration time is excluded. OS page caches and the process-global native
dictionary remain uncontrolled or warm. This is connection-cache sensitivity,
not a measurement of a physically cold disk or first-ever Jieba startup.

This is a synthetic FTS workload with small vectors, not a service SLA or an
end-to-end mine/why benchmark. CPU frequency, WSL scheduling and background
host load are uncontrolled. Larger embedding dimensions, real document
distribution, storage devices and concurrent writers can change absolute
latency. All measurements keep the detail feature explicitly on demand.

## Observed results

Environment: stable rustc 1.98.1 (48a229cea 2026-09-01), cargo 1.98.1
(797e8a9bc 2026-08-05), SQLite 3.46.0, Fedora 44 Linux x86_64 / WSL2
kernel 6.18.40.1-microsoft-standard-WSL2, Intel Core i7-12700H, six virtual
CPUs. Disk fixtures used the workspace Linux filesystem, not /tmp tmpfs.
The complete matrix finished in 206.36 seconds with 288 measurement cases:
192 warm cases and 96 independently reopened disk-connection cases.
No competing build or test job ran during the core timing matrix.

All sampled pipeline results preserve the existing score explanation, phrase
scoring facts and original-text ranges. The maximum BM25 contribution-sum
error was 1.7763568394002505e-15. All complete pipelines
send exactly one fixed query embedding; mapping alone sends none.

The following medians are milliseconds for the frequent common query on the
57-byte short target, a warm disk connection, top-k 10 and the fixed hybrid
plan. The last column uses the longer alpha beta common 知识 中国 query and
shows complete minus baseline, not just mapping.

| Chunks | Strategy | Baseline | Diagnostics, no mapping | Complete | Mapping only | Long query extra |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | unicode61 | 0.526 | 0.549 | 0.556 | 0.013 | 0.297 |
| 1,000 | trigram | 0.834 | 1.054 | 1.046 | 0.022 | 1.265 |
| 1,000 | simple | 0.619 | 0.761 | 0.789 | 0.017 | 0.820 |
| 1,000 | jieba | 0.652 | 0.807 | 0.817 | 0.021 | 1.052 |
| 10,000 | unicode61 | 4.447 | 4.347 | 4.483 | 0.024 | 1.101 |
| 10,000 | trigram | 6.215 | 7.178 | 7.287 | 0.037 | 6.003 |
| 10,000 | simple | 5.315 | 5.739 | 5.822 | 0.040 | 5.287 |
| 10,000 | jieba | 5.472 | 5.669 | 5.913 | 0.045 | 6.507 |
| 100,000 | unicode61 | 48.959 | 48.244 | 48.063 | 0.039 | 13.192 |
| 100,000 | trigram | 60.374 | 70.852 | 70.634 | 0.046 | 57.335 |
| 100,000 | simple | 51.238 | 56.774 | 55.876 | 0.053 | 42.863 |
| 100,000 | jieba | 51.465 | 56.546 | 55.875 | 0.058 | 53.899 |

Rare-query and long-target measurements are also included in the raw report.
Across the warm disk cases, mapping-only medians range from 0.010 to 0.076 ms
for the short target and from 0.420 to 2.451 ms for the 24,586-byte target.
The longest target/query combination has 12,288 hits: full output and
all range allocations are retained without truncation.

At 100k rows, the short-target long-query diagnostic adds 13.192–57.335 ms
over baseline. This is materially larger than the historical tiny auxiliary
probe and includes the production candidate pipeline, corpus phrase scans
and transport. Rare/frequent distributions and actual phrase counts matter;
there is no fixed overhead bound. Original-text mapping remains a smaller
component in this matrix. No retrieval, BM25 or fusion optimization was
justified or applied to change these semantics.

48 of the 288 complete-minus-no-mapping median differences were negative;
this exposes sampling/scheduling noise, especially alongside expensive
candidate retrieval. It does not imply that adding mapping makes the pipeline
faster. Use the separate mapping-only observations for that component.

Reopening is also sensitive to caches. For the 100k common-query short target:

| Strategy | Warm complete ms | Reopened first-query complete ms |
| --- | ---: | ---: |
| unicode61 | 48.063 | 49.112 |
| trigram | 70.634 | 69.857 |
| simple | 55.876 | 56.339 |
| jieba | 55.875 | 58.805 |

Public lexical_details adapter plus JSON byte encoding, measured separately:

| Target bytes | Hits | JSON bytes | Median ms |
| ---: | ---: | ---: | ---: |
| 18 | 2 | 794 | 0.007 |
| 2,304 | 256 | 40,930 | 0.585 |
| 18,432 | 2,048 | 329,116 | 5.010 |

These are 21 samples per unicode61 payload. The outer score/provenance JSON
and output I/O are excluded. High hit counts can grow output size and adapter
cost independently of the corpus scans.

Raw reports and all samples are retained locally under
.ai/process/fts-bm25-eval/20261003-loop5/performance.json and public-json.json.
The commands above regenerate equivalent synthetic fixtures; timing values
will vary. Keep --fts-details explicit and use the result as diagnostic
evidence rather than a production latency promise.

## Native platform validation

The Rust CI workflow already runs full tests and Clippy on ubuntu-latest,
macos-latest and windows-latest. The new normal FTS regressions participate
in that matrix; opt-in timing tests remain ignored.

This working tree was tested locally on Linux x86_64 / WSL2. No macOS executor
or usable Windows Rust toolchain is available in this session. No commit,
push or remote CI run was made for these changes. Native macOS/Windows
execution therefore remains pending; configured CI and cross-compilation
are not evidence of native test success.

Before closing platform validation, run the repository Rust gates on the
reviewed revision and record the revision and actual CI results. All normal
tests cover DOCUMENT callback errors and ownership, four query strategies,
UTF-8/pinyin/colocated/prefix mappings, stale text, WAL snapshots and isolated
Jieba-cache initialization. The current session uses cargo test; cargo-nextest
is not installed locally.
