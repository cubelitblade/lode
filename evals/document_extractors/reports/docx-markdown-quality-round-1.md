# DOCX canonical Markdown quality round 1

Date: 2026-09-12 04:22 UTC
Run: `20260912T042243Z-docx-markdown-quality`

## Scope and method

This round evaluates the production `lode-core` DOCX adapter after the
canonical Markdown switch. Five generated fixtures have manually defined
heading, segment, and plain-content truth. The 34 pinned Apache POI samples are
compared against the existing Python extractor only for content projection and
robustness diagnostics; they do not provide Markdown structural truth.

The Markdown projection removes generated heading/list/table syntax, emphasis,
links, and tags for content comparison. Raw Markdown is separately checked for
delimiter parity and lossless segment joining.

## Results

| Fixture | Projected F1 | Anchor order | Boundaries | Provenance | Headings | Delimiters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `structured` | 0.958 | 1.000 | 0.667 | 0.000 | 0.000 | PASS |
| `split-runs` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| `heading-hierarchy` | 0.972 | 1.000 | 0.000 | 0.000 | 0.000 | PASS |
| `interleaved-tables` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| `text-features` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |

Annotated Markdown gate: **FAIL**.

Public corpus diagnostics:

- Extraction success: 34/34 (`1.000`).
- Mean projected F1 against Python baseline: `0.942`.
- Delimiter parity: `33/34` balanced; `public-025` is a conservative diagnostic failure.
- Lowest projected F1: `public-024` (`0.511`), `public-003` (`0.565`), `public-019` (`0.687`).

## Findings

`Title`/Word `Heading 0` is not recognized as an IR heading by
`office_oxide 0.1.10`. In `structured` and `heading-hierarchy`, the title is
therefore emitted as a paragraph, while the first recognized heading is also
used as a synthetic section title. This causes a missing title heading,
duplicate heading content, and incorrect Segment boundaries/provenance.

The ordinary Heading 1–3, inline formatting, table, mixed-script, and segment
joining checks pass. The title handling must be resolved before the canonical
Markdown path can pass Gate 2. The public corpus needs a later manually
adjudicated Markdown truth set; its current scores are regression diagnostics,
not a release gate.

The complete machine-readable report is kept under the locally excluded
`.ai/process/extractor-evals/20260912T042243Z-docx-markdown-quality/` directory.
