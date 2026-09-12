# PDF extractor quality round 2

Date: 2026-09-11
Run: `20260911T063943Z-pdf-quality-2`

## Outcome

`pdf_oxide 0.3.78` does not pass the focused quality gate under the frozen NFC
contract. It passes one of four manually adjudicated selection-scope samples;
PyMuPDF passes two. This reverses neither round 1's complete valid-file success
nor its generated-fixture results, but demonstrates that baseline agreement
hid three concrete extraction limitations.

No candidate is promoted to Operational by this round.

## Manual adjudication

Six disputed public PDFs were rendered and inspected. Their byte identities,
visible content, expected order, per-page anchors, scope classification, and
review notes are frozen in `truth/pdf-quality-round-2.json`.

- Selection scope: landscape-page order, two Arabic CID font samples, AcroForm.
- Diagnostic scope: a 14-page table-like standard-font grid and an intentionally
  invalid veraPDF ToUnicode/formula sample.
- NFC remains the planned quality contract. NFKC is an additional diagnostic,
  not a replacement gate.

## Summary

| Candidate | Selection pass | Diagnostic pass | Strict mean F1 | Compatibility mean F1 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Python/PyMuPDF | 2/4 | 0/2 | 0.384 | 0.667 | FAIL |
| `pdf_oxide` | 1/4 | 0/2 | 0.605 | 0.664 | FAIL |

## Findings

- Arabic CID TrueType: `pdf_oxide` reaches NFKC F1 and order `1.000`, but NFC
  F1 is `0.823` because its output mixes ordinary Arabic code points and
  presentation forms. PyMuPDF has the same compatibility problem more
  severely. An explicit normalization decision could solve this case.
- Arabic CID Unicode-map: the rendered page visibly contains `عربي`.
  `pdf_oxide` returns no text, while PyMuPDF returns four `U+FFFF`
  noncharacters. Both are genuine failures, not reference ambiguity.
- Rotation: `pdf_oxide` retains nearly all text (F1 `0.993`) but reverses the
  two visible lines on the landscape page. Plain text and generated Markdown
  both score `0.750` for order.
- AcroForm: both candidates retain the visible `TextField:` label and pass all
  manual anchors. They differ only in whitespace and footer placement.
- Standard-font grid: `pdf_oxide` preserves all 14 page anchor groups, versus
  `0.976` for PyMuPDF, and emits fewer unsafe controls (`138` versus `510`).
  Neither output is terminal-safe without sanitization. Grid reconstruction is
  non-gating because it is table-like.
- Invalid ToUnicode formula: neither candidate produces a satisfactory linear
  form. PyMuPDF preserves prose spacing better; `pdf_oxide` loses spaces and
  rearranges formula tokens. This remains a non-gating malformed-input limit.

## Markdown implications

Markdown mirrors the underlying extraction behavior: it cannot repair missing
Arabic text or landscape reading order. Its delimiter parity remains clean,
but the standard-font output contains inferred headings, code fences, links,
and style markers, so the CLI must treat it as an optional display view and
sanitize controls before rendering. Canonical indexing should remain page-level
plain text plus outline provenance.

## Review decision required

The predefined gate does not permit advancing `pdf_oxide` directly. The next
reviewable choices are:

1. Run a bounded remediation spike for NFKC post-normalization and landscape
   line ordering, then repeat these six cases. The missing Arabic Unicode-map
   sample may still require an upstream fix or remain an explicit limitation.
2. Accept these failures by revising the supported scope and advance
   `pdf_oxide` to Operational.
3. Follow the original fallback and qualify a statically linked
   `pdfium-render` candidate before another quality comparison.

No production dependency or extractor implementation is authorized by this
report.
