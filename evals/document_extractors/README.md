# Document Extractor Evaluation

This experiment compares document extraction implementations behind a common
JSON protocol. Selection rounds use the standalone `rust-runner`; Operational
rounds invoke the selected production `lode-core` adapter through the minimal
`production-runner` crate.

## Implemented rounds

### Legacy DOC Smoke

```bash
uv run python -m evals.document_extractors.run --format doc --round smoke
```

This is the first gate for Word 97–2003 binary `.doc` support. It runs the
independent `office_oxide` and `rwml` adapters against the pinned, license-clean
synthetic corpus in `corpora/doc-rwml.json`, plus corrupt and truncated inputs.
The gate checks build and license eligibility, valid-file parsing, anchor order,
and explicit malformed-input rejection. It does not change or invoke the
production `lode-core` extractor.

The frozen boundary is recorded in
[`reports/doc-smoke-contract.md`](reports/doc-smoke-contract.md).

The Quality comparison is:

```bash
uv run python -m evals.document_extractors.run --format doc --round quality
```

It applies stricter text, anchor, segment-boundary, malformed-input, and table
presence checks to the same pinned corpus. Markdown remains a structural
diagnostic only.

The first result is summarized in
[`reports/doc-quality-round-1.md`](reports/doc-quality-round-1.md). That report
is historical: it used the candidate-owned `rwml` corpus and evaluated
`office_oxide`'s format-specific Markdown path. The current production
decision uses `office_oxide::plain_text()` for DOC, while the shared IR path
remains a later structural-table experiment.

### DOCX smoke and quality

```bash
uv run python -m evals.document_extractors.run --format docx --round smoke
```

Candidate builds use the repository Rust toolchain by default. Set
`LODE_EVAL_RUST_TOOLCHAIN=<toolchain>` only to reproduce a run with a specific
installed toolchain; reports record the actual compiler version.

The smoke gate checks that every candidate:

- builds with the repository Rust toolchain and has an allowed license;
- accepts document bytes rather than requiring an external service;
- parses the generated valid fixtures;
- retains manually selected anchors in reading order;
- rejects a corrupt OOXML package without panicking or timing out.

Reports are written under `.ai/process/extractor-evals/`, which is locally
excluded from Git. Smoke results establish feasibility only; selection happens
after the quality and operational rounds.

The frozen first DOCX quality result and provisional selection are summarized
in [`reports/docx-quality-round-1.md`](reports/docx-quality-round-1.md).

The canonical Markdown quality round evaluates the production DOCX path with
heading, segment, projection, delimiter, and public-corpus diagnostics:

```bash
uv run python -m evals.document_extractors.run --format docx --round quality-markdown
```

### PDF smoke

```bash
uv run python -m evals.document_extractors.run --format pdf --round smoke
```

The PDF smoke round checks page segmentation, blank pages, outline provenance,
rotated text, image-only pages, and explicit rejection of encrypted and corrupt
inputs. Two-column and table fixtures are diagnostic known-limit samples and do
not affect selection gates. The round also records whether a candidate emits
Markdown and performs a narrow delimiter-parity check relevant to a future CLI
token-completion strategy. Markdown quality is diagnostic in Smoke and must be
evaluated with richer fixtures in Quality.

The first result is summarized in
[`reports/pdf-smoke-round-1.md`](reports/pdf-smoke-round-1.md).

### PDF quality

```bash
uv run python -m evals.document_extractors.run --format pdf --round quality
```

This round compares Python/PyMuPDF and `pdf_oxide` on exact generated truth,
pinned PDF.js and veraPDF subsets, and optional locally excluded private files.
Plain-text selection metrics and Markdown diagnostics are reported separately.
The first result is summarized in
[`reports/pdf-quality-round-1.md`](reports/pdf-quality-round-1.md).

Disputed font, rotation, form, and malformed Unicode-map samples have a second,
manually adjudicated round:

```bash
uv run python -m evals.document_extractors.run --format pdf --round quality-2
```

Its truth file pins every reviewed PDF by SHA-256 and scores the rendered text
against human-defined content and page anchors. NFC remains the selection
contract; NFKC is reported only as a compatibility diagnostic. The result is
summarized in
[`reports/pdf-quality-round-2.md`](reports/pdf-quality-round-2.md).

A bounded runner-only remediation experiment is summarized in
[`reports/pdf-quality-remediation-spike.md`](reports/pdf-quality-remediation-spike.md).

### Operational

Operational measurements run the selected Rust candidate in release mode with
the repository stable toolchain, a 15-second per-file timeout, repeated fixture
extraction, and best-effort peak RSS measurement on POSIX hosts:

```bash
uv run python -m evals.document_extractors.operational --format docx
uv run python -m evals.document_extractors.operational --format pdf
uv run python -m evals.document_extractors.operational --format doc \
  --public-corpus-dir /path/to/rwml/corpus/public/benchmark/sample
```

The report records the exact compiler version, p50/p95 latency (three
iterations by default), runner and production build time, dependency count,
release binary size, and invalid-input behavior. Valid layout samples outside the
selected PDF scope (for example, two-column pages) remain visible as
diagnostics but do not affect the gate. Reports are written to
`.ai/process/extractor-evals/`.

The current production `.doc` run is summarized in
[`reports/doc-operational.md`](reports/doc-operational.md). The current
implementation uses `office_oxide 0.1.10`; the former `rwml 0.1.4` run is
historical only. A manual production-binary `mine -> prospect -> dig` check
against an independent public `.doc` sample also passed. Cross-platform
release CI remains before final cutover.
