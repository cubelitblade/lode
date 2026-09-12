# Legacy DOC Smoke contract

This contract freezes the first gate for Word 97–2003 binary `.doc` support.
It is an evaluation boundary, not a production support claim.

## Scope

- Input: OLE2/CFB Word binary documents with a `.doc` suffix.
- Candidates: isolated native Rust adapters for `office_oxide = 0.1.10` and
  `rwml = 0.1.4`.
- Valid truth: three MIT, repository-owned synthetic samples from the pinned
  `rwml` public benchmark manifest.
- Required behavior: parse valid documents, retain selected text anchors in
  reading order, and expose one usable text segment.
- Negative truth: a non-OLE2 file and a truncated OLE2 file must return an
  extraction error without a crash or timeout.

## Gate

A candidate is Smoke-pass only if all of these hold:

1. The standalone runner builds with the repository Rust toolchain.
2. Direct and transitive dependency license checks pass.
3. All valid samples parse successfully.
4. All malformed samples are rejected.
5. The minimum anchor-order score is `1.000`.

Smoke does not select a production adapter. Quality must still evaluate richer
paragraph/table structure and corpus behavior. In the first Quality slice,
`nested_tables.doc` checks table presence as a structural diagnostic; it does
not claim full nested-table reconstruction. Operational validation must still
run through the production boundary.

Production `lode-core` integration is governed by those later gates; this
Smoke contract alone does not make a production support claim.
