# PDF Operational round

Run `20260912T010302Z-pdf-operational` used the production `lode-core`
adapter, Rust 1.88 release builds, three iterations per fixture, and a
15-second per-file timeout.

- Linux/WSL2; runner build 1.93s; production workspace build 0.24s
- Runner dependencies: 328 packages; production dependencies: 339 packages
- Runner binary: 10,731,840 bytes
- Production binaries: `lode` 23,147,200 bytes; `lode-mcp` 416,320 bytes
- Throughput: 1,619,639 bytes/s
- Gate: **PASS** (all in-scope valid/invalid fixtures passed frozen text,
  heading, page, form, and error truth)
- Two-column fixture remains visible as an excluded diagnostic and is not part
  of the gate.

No pre-dependency binary artifact was available, so a size delta is not
claimed. macOS/Windows builds remain CI gates.
