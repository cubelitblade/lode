# PDF extractor smoke round 1

Date: 2026-09-11
Run: `20260911T051918Z-pdf-smoke`

## Outcome

`pdf_oxide 0.3.78` passes Smoke and advances to PDF Quality. `pdf-extract
0.12.0` does not advance in its current adapter because it accepts an encrypted
PDF as a successful empty document. The Python/PyMuPDF baseline passes Smoke.

`pdfium-render` was not admitted: the required Linux, macOS, and Windows static
Pdfium artifact setup has not yet been demonstrated.

## Fixtures and gate

Six generated fixtures cover five pages with a nested outline and a blank page,
two-column text, a rotated page, an image-only page, password encryption, and a
corrupt cross-reference table. Smoke requires all valid fixtures to parse,
encrypted and corrupt inputs to return errors, allowed licenses, and no crash or
timeout. The two-column fixture is a non-gating diagnostic; aggregate order,
boundary, and provenance scores include only the supported selection scope.

## Results

| Candidate | Valid | Invalid rejected | Min order | Boundaries | Provenance | Markdown coverage | Delimiter parity | Smoke |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Python baseline | 4/4 | 2/2 | 1.000 | 1.000 | 0.917 | 0.000 | N/A | PASS |
| `pdf_oxide` | 4/4 | 2/2 | 1.000 | 1.000 | 1.000 | 1.000 | PASS | PASS |
| `pdf-extract` | 4/4 | 1/2 | 1.000 | 0.750 | 0.667 | 0.000 | N/A | FAIL |

`pdf_oxide` produced exact page numbers and persistent outline heading chains,
including inheritance onto a continuation page and across a blank page. The
Python baseline only applies an outline heading on its starting page, which
accounts for its lower provenance result. `pdf-extract` exposes page text but no
outline in the evaluated high-level API.

## Reading order and Markdown finding

The initial default-options run misclassified the synthetic two-column page as
a table and duplicated content. The accepted conservative adapter now sets
`extract_tables = false` for both plain text and Markdown. This removes the
false table and duplicate, but output remains row-interleaved (`Left one`,
`Right one`, then the second row), confirming that general column ordering is
not cheaply or reliably solved. The sample remains visible as a diagnostic and
does not affect selection.

`pdf_oxide` nevertheless has the most useful interface for the planned CLI: it
can return plain page text for indexing, page-level Markdown for display, and
outline data for heading provenance. The current Lode `Segment` stores only
plain text, heading, and page, so Markdown must remain a separate optional view
or require a richer segment model. It should not replace canonical extraction
text until the Quality round validates its content and layout behavior.

## Scope decision and PDF Quality focus

- Multi-column reading order and PDF table reconstruction are out of scope and
  explicitly not guaranteed. Retain them only as non-gating diagnostics.
- Score supported single-flow plain text and Markdown separately for omissions,
  duplication, order, and heading inference.
- Keep `extract_tables = false`; do not synthesize Markdown tables from PDF
  geometry.
- Add nested and malformed outlines, named destinations, CJK fonts, ligatures,
  forms, and incremental/cross-reference variants.
- Run the pinned PDF.js and veraPDF public corpora, plus optional private
  documents, and retain per-sample bounded diffs.
- Keep encrypted-file classification as a hard gate and image-only PDFs as a
  successful empty extraction.
