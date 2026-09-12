# DOCX Operational round

Run `20260912T005630Z-docx-operational` used the production `lode-core`
adapter, Rust 1.88 release builds, three iterations per fixture, and a
15-second per-file timeout.

- Linux/WSL2; runner build 0.20s; production workspace build 0.42s
- Runner dependencies: 328 packages; production dependencies: 339 packages
- Runner binary: 10,731,832 bytes
- Production binaries: `lode` 23,147,200 bytes; `lode-mcp` 416,320 bytes
- Throughput: 20,087,545 bytes/s
- Gate: **PASS** (all six fixtures, including corrupt OOXML, passed status and
  frozen segment truth)

No pre-dependency binary artifact was available, so a size delta is not
claimed. macOS/Windows builds remain CI gates.
