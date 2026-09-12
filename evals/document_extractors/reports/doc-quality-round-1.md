# Legacy DOC quality round 1

Date: 2026-09-12

## Scope

This round compares the isolated `office_oxide 0.1.10` and `rwml 0.1.4`
adapters against the pinned MIT `rwml` legacy DOC corpus. It includes three
valid synthetic documents and the two malformed inputs used by Smoke. The
production `lode-core` adapter is not involved.

The quality gate requires complete valid-file success, complete malformed-file
rejection, minimum 3-gram F1 `0.98`, minimum anchor order `0.97`, segment
boundary accuracy `0.99`, exact table-presence truth, an allowed license, and
zero crashes or timeouts.

## Result

| Candidate | Valid | Invalid rejected | Min F1 | Min order | Boundaries | Table structure | Score | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `office_oxide 0.1.10` | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 | 0.667 | 83.33 | FAIL |
| `rwml 0.1.4` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 | PASS |

`office_oxide` extracts the table's text but emits it as tab-separated content
and does not expose a table-shaped Markdown view in this runner. `rwml` emits
a table-shaped Markdown view and preserves the expected segment boundary for
the fixed nested-table sample. The nested table truth checks table presence,
not full nested-table reconstruction.

## Decision

`rwml 0.1.4` is the provisional candidate for the next Operational round.
This is not yet a production support decision: Operational must validate the
production boundary, release behavior, latency, memory, and cross-platform
build gates before `.doc` is enabled in `lode-core`.

The full machine-readable result is locally available at
`.ai/process/extractor-evals/20260912T141357Z-doc-quality/report.json`.
