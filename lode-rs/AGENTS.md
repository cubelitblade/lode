# AGENTS.md

Guidelines for AI coding agents working in `lode-rs`.

This directory contains the rust version of this repository.
It would replace the Python one on `v0.1.0-alpha.3`.

Common commands:

```bash
cargo check
cargo fmt
cargo test
```

Before commit:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```
