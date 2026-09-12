# DOCX canonical Markdown operational round

Run `20260912T055456Z-docx-operational` used the production `lode-core` DOCX
adapter, release builds, and three iterations per fixture with a 15-second
per-file timeout.

- Linux/WSL2; runner build `0.27s`; production workspace build `0.42s`
- Compiler: stable `rustc 1.98.1`; the exact version is retained as run
  provenance
- Runner dependencies: 331 packages; production dependencies: 342 packages
- Runner binary: `13,329,208` bytes
- Production binaries: `lode` `25,623,736` bytes; `lode-mcp` `451,440` bytes
- Throughput: `9,990,878` bytes/s
- Peak RSS: approximately `5,184 KiB` across the generated fixtures
- Gate: **PASS** (all six fixtures, including corrupt OOXML, passed canonical
  semantic truth and status checks)

Canonical Markdown truth is compared through a Markdown content projection,
while Segment heading and page provenance remain exact. No fixed performance or
binary-size threshold is imposed in this round; macOS/Windows builds remain CI
gates.
