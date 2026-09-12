# Document extractor selection summary

Date: 2026-09-13

## Current selection

| Format | Provisional choice | Scope/status | Next gate |
| --- | --- | --- | --- |
| DOCX | `office_oxide 0.1.10` | Quality and restricted-scope Operational pass; ordinary paragraphs/tables and heading chains | Cross-platform CI |
| PDF | `pdf_oxide 0.3.78` | Restricted single-flow digital PDF Quality and Operational pass | Cross-platform CI |
| DOC | `office_oxide 0.1.10` | Independent-corpus validation pass and production CLI integration check for conservative plain text; table IR deferred | Cross-platform CI |

The PDF choice explicitly excludes general multi-column reading order, PDF
table reconstruction, OCR/scanned text, and general RTL PDFs whose glyph-to-
Unicode mapping or logical order is unreliable. This is a scope decision, not
an assertion that `pdf_oxide` solves those cases.

MCP remains the final stage and is not part of this selection.

## Evidence

### DOCX

`office_oxide 0.1.10` passed the generated structural truth, all 34 public
Apache POI samples, and the crash/timeout gate. It scored `98.13`; `rwml`
scored `96.83`. `rs-docx` was rejected despite exact ordinary table rows:
strict XML failures and a deeply nested table caused a process-level stack
overflow, and one fixture lost a tab.

Report: [`docx-quality-round-1.md`](docx-quality-round-1.md)

### PDF

`pdf_oxide 0.3.78` passed the broad supported-scope Quality round with 21/21
public files parsed and no crashes or timeouts, but its public baseline mean F1
was `0.882`. Manual adjudication found real failures in Arabic CID Unicode
mapping and one landscape reading order.

The runner-only remediation prototype combined NFKC, RTL-preserving assembly,
and geometric line ordering. It improved the focused selection result from
`1/4` to `3/4`:

- Arabic presentation-form sample: fixed;
- landscape line order: fixed;
- Arabic CID sample with missing mapping: still empty;
- Markdown RTL order: still incorrect.

Reports: [`pdf-quality-round-1.md`](pdf-quality-round-1.md),
[`pdf-quality-round-2.md`](pdf-quality-round-2.md), and
[`pdf-quality-remediation-spike.md`](pdf-quality-remediation-spike.md)

Restricted-scope Operational reports: [`docx-operational.md`](docx-operational.md)
and [`pdf-operational.md`](pdf-operational.md).

### DOC

The former `rwml 0.1.4` result passed a pinned three-file legacy DOC Quality
round, but that corpus was candidate-owned and the production path flattened
text. It is retained as historical evidence, not as the current selection
gate.

The revised decision uses `office_oxide 0.1.10` for DOC plain text, matching the
existing DOCX production dependency. Validation against eight independently
sampled Apache POI and LibreOffice legacy-DOC fixtures produced text/status
behavior consistent with the former path: six samples had indexable text and
two explicitly had none; corrupt and truncated inputs were rejected. This is a
validation slice, not yet a replacement statistical Quality round. The
`office_oxide` shared IR path remains deferred because one independent sample
lost its first paragraph under that projection.

Reports: [`doc-smoke-contract.md`](doc-smoke-contract.md),
[`doc-quality-round-1.md`](doc-quality-round-1.md), and
[`doc-operational.md`](doc-operational.md)

`pdf-extract 0.12.0` was rejected at Smoke because an encrypted PDF was
accepted as a successful empty document. `pdfium-render` was not qualified
because the required three-platform static setup was not demonstrated.

## Format and presentation contract

- Canonical indexing output remains plain text segments with page provenance.
- PDF uses one segment per non-empty page and outline-derived heading
  provenance.
- Markdown is an optional CLI display view, not extraction truth. Symmetric
  token completion can repair a display truncation, but cannot restore missing
  glyph mappings or lost structure.
- DOCX flattened segments are easy for the CLI to consume, but currently lose
  inline formatting, links, list semantics, and rich table structure.

## Known limitations and risks

- Public-corpus comparisons against PyMuPDF are useful regression signals, not
  universal semantic truth.
- No private real-document corpus was available in these rounds.
- DOCX Operational is complete on Linux/WSL2 with Rust 1.88; three-platform
  builds and final dependency audit remain CI responsibilities.
- PDF restricted-scope Operational is complete on Linux/WSL2 with Rust 1.88;
  three-platform builds and final dependency audit remain CI responsibilities.
- DOC Operational is complete with the current `office_oxide` production
  adapter. The independent validation corpus remains a small diagnostic slice,
  and three-platform builds plus final dependency audit remain CI
  responsibilities.
- Missing PDF `ToUnicode` mappings are information loss, not merely a bidi
  sorting defect. Adobe's PDF reference describes `ToUnicode` as the mapping
  from character codes to Unicode and notes that without it some glyphs have no
  recoverable character meaning:
  [PDF Reference](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.3.pdf).
- Unicode bidi processing operates on Unicode logical text and controls its
  display order; it cannot reconstruct characters that the PDF never mapped:
  [Unicode UAX #9](https://www.unicode.org/reports/tr9/).
- Formula linearization, font grids, complex mixed-direction layouts, OCR,
  exact pagination, embedded objects, and macros remain outside this phase.

## Decision record

The current decision is to proceed with `office_oxide` for both DOCX and
legacy DOC, using DOCX's existing IR projection and DOC's conservative
`plain_text()` projection. `pdf_oxide` remains selected for the deliberately
narrowed PDF scope. DOC Operational and a manual production-binary
`mine -> prospect -> dig` integration check have passed; cross-platform CI
remains before release. The integration check is evidence for this decision,
not yet an automated regression test. Dependency license and vulnerability
checks remain final gates.
