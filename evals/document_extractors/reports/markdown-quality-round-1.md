# Native Markdown quality round 1

Date: 2026-09-12

This report records the checks actually run for the native Markdown Gate 1–3
implementation. The Rust workspace gates and the in-process Markdown quality,
storage-identity, incremental-sync, and CLI process-level matrix ran on Linux;
cross-platform CI remains a release gate.

## Results

| Metric | Result |
| --- | --- |
| Source preservation | PASS for every exercised non-whitespace Markdown fixture, including YAML metadata, nested containers, CRLF, UTF-8 BOM, and UTF-16 decoding |
| Boundary accuracy | PASS for the exercised ATX, Setext, nested blockquote/list, fenced code, footnote, and empty-heading cases |
| Provenance accuracy | PASS for the exercised heading chains and formatted visible heading text |
| Extractor identity | PASS in the targeted store case: different chunks create separate `text` and `markdown` rows; `.md` and `.markdown` reuse |
| Incremental behavior | PASS: `.md` to `.markdown` rename reuses content without embedding; `.txt` to `.md` re-extracts and re-embeds |
| Schema rebuild signal | PASS in the targeted store test for schema version 2 rejection |
| Raw CLI flow | PASS: `mine -> prospect -> dig --view json` retained the `.md` path, heading provenance, and raw Markdown text |

## Verification

Executed on `x86_64-unknown-linux-gnu` with repository stable toolchain
`rustc 1.98.1 (48a229cea 2026-09-01)`:

```text
cargo fmt --all --check
cargo test -p lode-core --lib                       # 225 passed
cargo test --workspace                              # passed
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo deny --locked check                           # passed with existing duplicate-version warnings
cargo audit                                         # passed with one allowed ttf-parser unmaintained warning
git diff --check
```

An initial clean-target compilation using
`CARGO_TARGET_DIR=/tmp/lode-rs-markdown-target` stopped because `/tmp` ran out
of space. The existing repository target was then used successfully. The
process-level CLI test was executed outside the filesystem sandbox because
the sandbox denies local TCP binds; its mock remained loopback-only and
hermetic. Cross-platform CI was not run in this round.
