# FTS5 Chinese Tokenizer Evaluation

## Purpose

This evaluation compares SQLite FTS5 tokenizer configurations for Chinese text
retrieval and determines a suitable default configuration for lode.

## Background

FTS5 tokenizers are responsible for splitting input text into searchable tokens.
SQLite provides built-in tokenizers such as `unicode61`, and also supports custom tokenizer implementations.

For Chinese text, tokenization strategy can significantly affect query matching behavior because words are not
separated by whitespace.

## Candidates

This evaluation focuses on tokenizers that are applicable to Chinese text.
Tokenizers designed primarily for non-Chinese text, such as `ascii` and `porter`, are excluded.

### unicode61

SQLite built-in tokenizer.

Tokenization is based on Unicode character categories and separators.
It does not perform language-specific word segmentation, which means
Chinese text is generally processed at the character level.

### trigram

SQLite built-in tokenizer.

Tokenization is based on fixed-length three-character sequences.
It does not require language-specific segmentation and can provide better
matching coverage for languages without explicit word boundaries.

### simple

Third-party tokenizer `simple`.

The tokenizer stores Chinese text using character-level indexing with additional
support for pinyin matching.

Unlike traditional word segmentation tokenizers, `simple` keeps the index
granularity independent from query segmentation.

### jieba

Third-party tokenizer `simple` with Jieba query strategy.

The query is segmented using Jieba and rewritten into an FTS5 `MATCH` expression.
It improves phrase-level matching and recall without changing the underlying
simple index.


## Evaluation

### Dataset

The evaluation uses [MTEB DuRetrieval](https://huggingface.co/datasets/mteb/DuRetrieval)
as the primary dataset.

Queries are grouped by character length into the following buckets:

- 1 character
- 2 characters
- 3 characters
- 4-9 characters
- 10+ characters
- All queries

Because public sentence-level retrieval benchmarks generally contain few short
substring queries, the 1-character, 2-character, and 3-character buckets are
augmented with synthesized substring probes.

Each probe is extracted from a labeled query and retained only when the substring
occurs in at least one relevant document. The original relevance judgments are
reused for these probes.

This reduces the bias toward longer queries and exposes failure cases caused by
insufficient token coverage or query processing limitations.

### Environment

#### Software

| Environment    | Version                                                    |
| -------------- | ---------------------------------------------------------- |
| OS             | Fedora Linux 44 (WSL2, Linux 6.18.33.2-microsoft-standard) |
| Python         | 3.13.14                                                    |
| SQLite runtime | 3.53.1                                                     |
| simple         | 0.7.1                                                      |

#### Hardware

| Hardware | Specification               |
| -------- | --------------------------- |
| CPU      | Intel(R) Core(TM) i7-12700H |
| Memory   | 16 GB                       |

### Metrics

The evaluation measures both retrieval effectiveness and index characteristics.

#### Retrieval Effectiveness

- **Recall@k**: measures the proportion of relevant documents retrieved within the top-k results.
- **Precision@k**: measures the proportion of retrieved documents that are relevant within the top-k results.
- **MRR@k**: measures how early the first relevant document appears in the top-k results.
- **Zero-hit rate**: measures the proportion of queries that return no results. It highlights tokenization strategies
   that fail to produce searchable matches for certain query types.

> [!NOTE]
> In this evaluation, `k` is fixed to 10.

#### Index Characteristics

- **Index size**: the size of the generated SQLite FTS5 database file after indexing the corpus.
- **Index build time**: measures the time required to create the FTS5 index from the corpus.

### Method

Each tokenizer candidate is evaluated by building an independent FTS5 index over the same corpus.

For each index tokenizer, all supported query strategies are evaluated using the corresponding index.
Each query retrieves the top-k documents, and the results are compared against the relevance judgments.

> [!NOTE]
> Only valid tokenizer and query-strategy combinations are evaluated.
> The `jieba` strategy reuses the `simple` index and does not require a separate index.

Retrieval metrics are aggregated by query-length bucket to analyze tokenizer behavior under different query lengths.

Index size, index build time, and evaluation time are recorded as additional engineering metrics.

## Results

The evaluation set holds 2,217 queries (200 labeled queries plus 2,017 synthesized probes),
bucketed as 1: 671, 2: 799, 3: 547, 4-9: 131, 10+: 69.

### Cost Summary

> [!IMPORTANT]
> Timings are medians over five consecutive runs in the single development environment
> described above (first-run cold-cache effects are reduced by reporting the median).
> They are not controlled absolute measurements: read the seconds as indicative only, and use the `×`
> factors — each column's cost relative to the fastest pairing — as the comparable signal.

| Index tokenizer | Query strategy | Index size | Build time    | Evaluation time |
| --------------- | -------------- | ---------- | ------------- | --------------- |
| unicode61       | unicode61      | 172.5 MiB  | 8.3 s (×1.0)  | 0.14 s (×1.0)   |
| trigram         | trigram        | 304.8 MiB  | 34.6 s (×4.2) | 0.33 s (×2.4)   |
| simple          | simple         | 269.9 MiB  | 22.0 s (×2.7) | 19.63 s (×140)  |
| simple          | jieba          | 269.9 MiB  | 22.0 s (×2.7) | 15.82 s (×113)  |

Both timing columns are normalized independently: build times run the same indexing job over
the same corpus, and evaluation times run the same query set, so each `×` factor provides a
relative estimate of the execution cost of that configuration.

The `simple` index is shared by the last two pairings, so both rows report the same
size and build time. Helper-backed queries (`simple` / `jieba`) cost about two orders
of magnitude more wall time than direct built-in tokenizer `MATCH` queries. This overhead
comes from additional query processing before execution, including query segmentation
and `MATCH` expression generation.

### Retrieval Effectiveness (top-k = 10)

#### Recall@10 by query-length bucket

| Pairing               | 1      | 2      | 3      | 4-9    | 10+    | All    |
| --------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| unicode61 + unicode61 | 0.0062 | 0.0152 | 0.0113 | 0.0148 | 0.0567 | 0.0128 |
| trigram + trigram     | 0.0000 | 0.0000 | 0.7256 | 0.3999 | 0.5818 | 0.2208 |
| simple + simple       | 0.0646 | 0.2509 | 0.4633 | 0.1931 | 0.1519 | 0.2404 |
| simple + jieba        | 0.0646 | 0.3765 | 0.5578 | 0.1503 | 0.1108 | 0.3052 |

#### Zero-hit rate by bucket

| Pairing               | 1      | 2      | 3     | 4-9   | 10+   | All   |
| --------------------- | ------ | ------ | ----- | ----- | ----- | ----- |
| unicode61 + unicode61 | 2.8%   | 48.1%  | 83.0% | 91.6% | 84.1% | 46.7% |
| trigram + trigram     | 100.0% | 100.0% | 0.0%  | 1.5%  | 0.0%  | 66.4% |
| simple + simple       | 0.0%   | 0.0%   | 0.0%  | 9.9%  | 31.9% | 1.6%  |
| simple + jieba        | 0.0%   | 0.0%   | 0.0%  | 44.3% | 62.3% | 4.6%  |

#### Precision@10 by bucket

| Pairing               | 1      | 2      | 3      | 4-9    | 10+    | All    |
| --------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| unicode61 + unicode61 | 0.0031 | 0.0049 | 0.0044 | 0.0061 | 0.0232 | 0.0049 |
| trigram + trigram     | 0.0000 | 0.0000 | 0.1916 | 0.1969 | 0.2290 | 0.0660 |
| simple + simple       | 0.0325 | 0.0852 | 0.1303 | 0.1031 | 0.0710 | 0.0810 |
| simple + jieba        | 0.0325 | 0.1213 | 0.1516 | 0.0809 | 0.0522 | 0.0973 |

Absolute Precision@10 values are affected by the number of relevant
documents available for each query. Since most DuRetrieval queries contain
only a small number of relevant documents, these values are mainly useful
for relative comparison between pairings.

#### MRR@10 by bucket

| Pairing               | 1      | 2      | 3      | 4-9    | 10+    | All    |
| --------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| unicode61 + unicode61 | 0.0127 | 0.0259 | 0.0304 | 0.0458 | 0.0966 | 0.0264 |
| trigram + trigram     | 0.0000 | 0.0000 | 0.6104 | 0.5707 | 0.5523 | 0.2015 |
| simple + simple       | 0.0668 | 0.2055 | 0.3699 | 0.4156 | 0.3551 | 0.2212 |
| simple + jieba        | 0.0668 | 0.2969 | 0.4545 | 0.3499 | 0.2657 | 0.2683 |

### Findings

- **simple + simple provides the most balanced trade-off in this evaluation**:
  it achieves the lowest overall zero-hit rate (1.6%) with no structural failure in any bucket.
- **jieba query strategy improves short-query retrieval but weakens on longer queries**
  **in this evaluation**: it achieves the best recall for 2-3 character queries
  (also the best short-query ranking, with MRR@10 of 0.30 at 2 characters and 0.45 at 3),
  while showing a higher zero-hit rate for 10+ character queries (62%).
- **trigram has a structural limitation below three characters**:
  a shorter query cannot produce a complete trigram token, so every 1-2 character query returns no results.
- **trigram performs best on 3-character queries in this evaluation**:
  a 3-character query maps to exactly one trigram, achieving high recall (Recall@10 = 0.73) with zero misses,
  and ranking its hits near the top (MRR@10 = 0.61).
- **trigram is the most precise matcher where it works**: on the buckets it can serve
  (3+ characters), its Precision@10 (0.19-0.23) runs 1.3-4.4× the helper-based pairings',
  with the gap widening as queries lengthen — trading coverage for cleaner result lists.
- **unicode61 provides limited Chinese retrieval effectiveness in this evaluation**:
  it achieves the lowest retrieval effectiveness among the tested configurations,
  despite having the smallest index and lowest execution cost.

## Conclusion

Based on the evaluation, `simple + simple` is selected as the default configuration.

The decision prioritizes robustness across query lengths over peak performance in individual buckets.
While `simple + jieba` achieves the highest overall retrieval effectiveness, it shows higher
zero-hit rates for longer queries. It remains a suitable optional strategy for workloads
dominated by short Chinese queries.

`trigram` is not selected as the default because its matching behavior has a structural
limitation for short queries: queries shorter than three characters cannot produce valid
trigram tokens. It remains useful for substring-oriented workloads where queries are typically
long enough to form trigrams, but it is not suitable as the general-purpose default.

`unicode61` is not selected because it does not provide sufficient Chinese term retrieval
coverage under this evaluation. It remains a reasonable choice for workloads with mostly
non-Chinese text or sparse Chinese content, where its lower index size and query overhead
are valuable.

From an engineering perspective, `simple + simple` also provides a
balanced resource profile. It requires less storage and build time than
`trigram`, while keeping query-side cost in the same order of magnitude
as `simple + jieba`. Since both strategies share the same index, users can
switch query processing strategies without rebuilding the index.

## Limitations

This evaluation uses a single Chinese retrieval benchmark and one development
environment. The timing results should be interpreted as relative comparisons
rather than absolute performance guarantees.

Synthesized probes test matching capability for short substrings,
but they do not model actual user intent or query frequency.
