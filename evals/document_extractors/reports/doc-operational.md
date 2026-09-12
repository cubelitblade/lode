# DOC Operational round

Run `20260912T142224Z-doc-operational` exercised the production `lode-core`
adapter with `rwml 0.1.4`, Rust 1.98.1, three iterations per fixture, and a
15-second per-file timeout.

- Linux/WSL2; runner build 174.34s; production workspace build 15.11s
- Runner dependencies: 333 packages; production dependencies: 344 packages
- Runner binary: 13,560,544 bytes
- Production binaries: `lode` 25,863,736 bytes; `lode-mcp` 451,440 bytes
- Throughput: 67,274,889 bytes/s
- Gate: **PASS** (three pinned valid public `.doc` fixtures and corrupt plus
  truncated invalid inputs)

All valid fixtures returned `ok` on every iteration and matched the frozen
truth anchors. Both malformed-input fixtures returned `error` on every
iteration without a crash or protocol failure. Per-fixture p50/p95 latency was
0.08/0.09 ms, 0.09/0.23 ms, and 0.08/0.40 ms for the valid fixtures; peak RSS
was 4,444–4,684 KiB. Invalid-input latency was 0.01/0.01–0.02 ms with peak
RSS of 4,412–4,452 KiB.

No pre-dependency production binary artifact was available, so a size delta is
not claimed. The current Operational gate has no fixed performance or binary
size threshold; it verifies parse/reject behavior, truth, and process
stability. macOS and Windows builds remain release CI gates.

The raw run is locally excluded from Git at
`.ai/process/extractor-evals/20260912T142224Z-doc-operational/report.md`.
