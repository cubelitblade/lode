# DOCX canonical Markdown quality round 2

Date: 2026-09-12 04:56 UTC
Run: `20260912T045616Z-docx-markdown-quality`

## Scope and method

This round repeats the production `lode-core` DOCX canonical Markdown gate
after adding explicit Word `Title` handling. A `Title` paragraph is promoted
to Markdown H1; ordinary Word `Heading 1`, `Heading 2`, and later headings are
shifted one level only when that document title exists. Synthetic section
titles are removed before rendering so they cannot duplicate body headings.

The five generated fixtures have manually defined heading, segment, and
plain-content truth. The 34 pinned Apache POI samples remain regression
diagnostics against the existing Python extractor; they do not provide
manually adjudicated Markdown structural truth.

## Results

| Fixture | Projected F1 | Anchor order | Boundaries | Provenance | Headings | Delimiters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `structured` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| `split-runs` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| `heading-hierarchy` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| `interleaved-tables` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| `text-features` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |

Annotated Markdown gate: **PASS**.

Public corpus diagnostics:

- Extraction success: 34/34 (`1.000`).
- Mean projected F1 against Python baseline: `0.945`.
- Delimiter parity: `33/34` balanced; `public-025` is a conservative
  diagnostic failure.
- Lowest projected F1: `public-024` (`0.511`), `public-003` (`0.565`),
  `public-019` (`0.687`).

## Findings

The annotated DOCX set now passes the complete Markdown gate. Documents with
a Word `Title` produce one H1 root title, with subsequent Word heading levels
shifted by one; documents without a `Title` retain the existing Heading 1 →
H1 mapping. Segment boundaries, provenance chains, inline formatting, tables,
mixed-script text, and lossless segment joining all pass.

The public corpus still needs a manually adjudicated Markdown truth set before
its diagnostic scores can become a release gate. `public-025` contains rich
content that triggers the intentionally conservative delimiter check; it does
not fail the annotated gate.

The complete machine-readable report is kept under the locally excluded
`.ai/process/extractor-evals/20260912T045616Z-docx-markdown-quality/` directory.
