# PDF extractor remediation spike

Date: 2026-09-11
Run: `20260911T143443Z-pdf-quality-2`

## Outcome

The isolated `pdf_oxide_remediated` runner candidate improves the focused
selection result from `1/4` to `3/4`, but it does not pass the production gate.
The remaining selection failure is the Arabic CID Unicode-map sample that
renders `عربي` but still produces empty extraction.

The prototype is not production code and has not been added to `lode-core`.

## Prototype changes

- Apply Unicode NFKC normalization to extracted plain text and Markdown.
- Use `pdf_oxide::extract_text_lines()` for the prototype's text assembly.
- Detect pages whose assembled text contains RTL scripts and keep the original
  assembler for those pages, because the line API returns Arabic in visual
  rather than logical order.
- Use the geometry-line output as the prototype Markdown page for non-RTL
  pages; keep the existing Markdown converter for RTL pages.

## Frozen six-sample result

| Candidate | Selection pass | Diagnostic pass | Strict mean F1 | Compatibility mean F1 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PyMuPDF | 2/4 | 0/2 | 0.384 | 0.667 | FAIL |
| `pdf_oxide` | 1/4 | 0/2 | 0.605 | 0.664 | FAIL |
| `pdf_oxide_remediated` | 3/4 | 0/2 | 0.667 | 0.667 | FAIL |

## Effect by sample

- Rotation: order `0.750 → 1.000`; text F1 `0.993 → 1.000`.
- Arabic CID TrueType: NFC F1 `0.823 → 1.000`; order `0.000 → 1.000`.
- Arabic CID Unicode-map: remains empty; no improvement.
- AcroForm: remains passing with all visible anchors.
- Standard-font grid: remains diagnostic; unsafe controls decrease from `138`
  to `136`, but are not eliminated.
- Invalid ToUnicode formula: remains diagnostic and does not gain a reliable
  formula linearization.

## Markdown finding

Canonical plain-text extraction benefits from the prototype. Markdown does not
yet have the same result for RTL pages: `pdf_oxide_remediated` still reports
Markdown anchor order `0.000` for the Arabic TrueType sample because the
library Markdown converter applies visual RTL ordering. Markdown must therefore
remain an optional display representation, with its own RTL-aware postprocess
or a fallback to canonical plain text. It cannot be used as the extraction
truth.

## Decision

The remediation is promising enough to inform a future adapter, but not enough
to select `pdf_oxide` yet. The unresolved Arabic CID mapping defect is a hard
selection-scope failure. The next choices are to investigate an upstream/font
mapping fix, explicitly narrow supported PDF text scope, or qualify a static
`pdfium-render` candidate. No production implementation is authorized by this
spike.
