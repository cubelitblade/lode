# DOC Operational round

Run `20260912T161111Z-doc-operational` exercised the current production
`lode-core` adapter with `office_oxide 0.1.10`'s legacy-DOC `plain_text()` path,
Rust 1.98.1, three iterations per fixture, and a 15-second per-file timeout.

- Linux/WSL2; runner build 11.09s; production workspace build 12.38s
- Runner dependencies: 331 packages; production dependencies: 342 packages
- Runner binary: 13,357,032 bytes
- Production binaries: `lode` 25,657,928 bytes; `lode-mcp` 451,440 bytes
- Throughput: 20,770,068 bytes/s
- Gate: **PASS** (three pinned valid public `.doc` fixtures and corrupt plus
  truncated invalid inputs)

All valid fixtures returned `ok` on every iteration and matched the frozen
truth anchors. Both malformed-input fixtures returned `error` on every
iteration without a crash, protocol failure, or timeout. Per-fixture p50/p95
latency was 0.28/0.33 ms, 0.28/0.28 ms, and 0.27/0.27 ms for the valid
fixtures; peak RSS was 4,952–5,288 KiB. Invalid-input latency was 0.23/0.33 ms
and 0.29/0.30 ms with peak RSS of 4,772–4,948 KiB.

The former `rwml 0.1.4` run remains historical evidence only; it used the
previous production dependency. No pre-dependency production binary artifact
was available, so a size delta is not claimed. The current Operational gate
has no fixed performance or binary-size threshold; it verifies parse/reject
behavior, truth, and process stability. macOS and Windows builds remain
release CI gates.

The raw run is locally excluded from Git at
`.ai/process/extractor-evals/20260912T161111Z-doc-operational/report.md`.
