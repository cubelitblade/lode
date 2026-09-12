# DOCX extractor quality round 1

Date: 2026-09-10
Run: `20260910T195423Z-docx-quality`

## Decision

Use `office_oxide 0.1.10` as the provisional DOCX selection. It is the highest
scoring candidate that passed every quality gate. This is not yet the production
selection: `office_oxide` must still pass the Operational round before it is
added to `lode-core`.

## Corpus and method

- Five generated, manually annotated fixtures cover heading chains, split runs,
  paragraph/table order, table rows, tabs, mixed scripts, and custom styles.
- Thirty-four public Apache POI DOCX files are pinned by revision and SHA-256 in
  `corpora/docx-apache-poi.json`.
- No private documents were available in this run.
- Generated fixtures use exact structural truth. Public-corpus content is
  compared with the Python baseline, so public F1 measures agreement rather
  than absolute correctness.

## Results

| Candidate | Annotated success | Min F1 | Boundaries | Provenance | Table rows | Public success | Public mean F1 | Score | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Python baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 | 1.000 | 99.85 | FAIL |
| `office_oxide` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.906 | 98.13 | PASS |
| `rwml` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.842 | 96.83 | PASS |
| `docx-rs` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 | 0.891 | 97.68 | FAIL |
| `rs-docx` | 1.000 | 0.940 | 0.800 | 1.000 | 1.000 | 0.912 | 0.924 | 96.14 | FAIL |

The Python baseline failed one public file because its XML parser exceeded the
configured depth. `office_oxide` and `rwml` parsed all public files; quality and
the smaller disagreement with the baseline put `office_oxide` first.

## `rs-docx` table finding

`rs-docx` extracted every row of the generated interleaved-table fixture
correctly (`1.000` table-row accuracy). Its overall table robustness was not
acceptable, however: two public documents failed on strict XML model
requirements (`Picture.fill` and `Table.grids`), and a deeply nested table-cell
document caused a process-level stack overflow. It also lost a tab in the
annotated text-features fixture. These failures exclude it despite good output
on ordinary tables.

## Markdown implications

The evaluation adapter deliberately normalizes DOCX into Lode's current
`Segment { text, heading, page }` contract. This shape is easy for the CLI to
render and safe to chunk, but it discards inline formatting, links, lists, and
true table structure. A later Markdown display feature should therefore render
from a richer internal representation or an explicitly stored Markdown view;
it should not try to reconstruct rich Markdown from the flattened text.

Symmetric token completion remains useful when a chunk begins or ends inside a
Markdown construct, but it cannot recover structure already discarded during
extraction. The Operational round should keep the plain-segment contract stable;
rich Markdown support can be designed separately without changing this
provisional selection.

## Next gate

Run DOCX Operational only after the PDF and DOC staged evaluations reach their
respective review points. Measure release p50/p95, throughput, RSS, dependency
count, build time, artifact delta, Rust 1.88 three-platform builds, and license
compliance.
