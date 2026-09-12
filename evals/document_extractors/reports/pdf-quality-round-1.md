# PDF extractor quality round 1

Date: 2026-09-11
Run: `20260911T062513Z-pdf-quality`

## Outcome

`pdf_oxide 0.3.78` passes the predefined supported-scope quality gate, but is
not yet promoted to Operational. Its generated-fixture structure is exact and
its public success rate is complete, while several font and formula samples
disagree materially with PyMuPDF. User review is required before either a
second Quality iteration or provisional selection.

## Corpus and scope

- Eleven generated fixtures: ten selection-scope cases and one non-gating
  two-column diagnostic.
- Thirteen PDF.js files pinned at
  `f4f90c2f6902fb2e707d1cc22c42fad4c7f67dc5`.
- Eight veraPDF files pinned at
  `01e40281d48e2f3755006fdf596ca25caaea8634`.
- No private PDF files were available.
- Multi-column reading order and PDF table reconstruction are explicitly out of
  scope. `extract_tables = false` is used for both plain text and Markdown.
- Public content metrics measure agreement with PyMuPDF, not independent
  semantic truth.

## Plain-text results

| Candidate | Annotated success | Min F1 | Min order | Boundaries | Provenance | Public success | Public mean F1 | Score | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Python/PyMuPDF | 1.000 | 1.000 | 1.000 | 1.000 | 0.969 | 1.000 | 1.000 | 99.61 | PASS |
| `pdf_oxide` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.882 | 97.64 | PASS |

`pdf_oxide` returned the expected status for all generated inputs, explicitly
rejected encrypted and corrupt files, parsed all 21 public files, produced one
segment per non-empty page, and applied outline headings persistently. There
were no crashes or timeouts.

## Material public disagreements

- `ArabicCIDTrueType.pdf`: F1 `0.208`, order `0.000`. `pdf_oxide` emitted more
  normalized Arabic code points but split and reordered part of the first line;
  PyMuPDF emitted Arabic presentation-form characters. NFC intentionally does
  not hide that difference.
- `arial_unicode_ab_cidfont.pdf`: F1 `0.000`; PyMuPDF returned four replacement
  characters while `pdf_oxide` returned no text. This comparison has no useful
  semantic truth and should not drive selection by itself.
- veraPDF Unicode-map failure sample: F1 `0.511`, order `0.200`; words lost
  spaces and the displayed formula was rearranged. This is a substantive
  extraction limitation even though the file is intentionally PDF/A-invalid.
- `standard_fonts.pdf`: F1 `0.855`. PyMuPDF exposes thousands of control and
  individual character lines, whereas `pdf_oxide` groups glyphs into readable
  rows; raw agreement understates the latter's readability.
- `rotation.pdf` and `file_pdfjs_form.pdf` retain nearly all content but differ
  in landscape-page order and form/footer placement.

These cases show that public-baseline F1 combines real extraction defects with
reference ambiguity. A second round should add manual anchors or truth only for
the disputed samples rather than tune against PyMuPDF blindly.

## Markdown diagnostics

| Candidate | Coverage | Mean precision | Mean F1 | Min order | Heading accuracy | Balanced |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Python/PyMuPDF | 0.000 | — | — | — | — | N/A |
| `pdf_oxide` | 0.960 | 0.886 | 0.889 | 0.000 | 0.714 | PASS |

Markdown was syntactically balanced for every emitted sample and was exact on
the supported generated text fixtures. The aggregate weaknesses come from the
same Arabic/font/formula inputs as plain text, plus heuristic heading inference:
visual font size can turn running headers or continuation lines into headings.

For the CLI, canonical indexing should remain page-level plain text with
outline-derived heading provenance. Generated Markdown may remain an optional
display view, but inferred headings should not replace outline metadata and the
CLI should retain symmetric token completion for chunk truncation.

## Review decision required

Choose one of the following before proceeding:

1. Run Quality round 2 with manual truth for the disputed Arabic, Unicode-map,
   standard-font, rotation, and form samples.
2. Accept the known font/formula limitations and provisionally advance
   `pdf_oxide` to Operational.

No production dependency or extractor implementation is authorized by this
report.
