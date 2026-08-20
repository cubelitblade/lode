# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project

`local-knowledge` — a local knowledge base semantic search tool exposed over MCP.
Currently in **early development**: only the embedding layer is implemented
(OpenAI-compatible HTTP client). The full vision and roadmap live in
[`.vscode/docs/PLAN.md`](.vscode/docs/PLAN.md) — read it before large changes,
and link to it instead of duplicating its content.

`README.md` and `main.py` are empty placeholders — do not treat them as
sources of truth.

## Build & Test

Managed with `uv` (Python >= 3.13, `src/` layout):

```bash
uv sync                 # install from uv.lock
uv run pytest           # run the test suite
uv run ruff check       # lint (run `ruff format --check` too)
uv run pyright          # strict type checking
```

## Architecture

Dependency inversion is the core pattern: business logic depends on
interfaces, never on concrete implementations.

- `src/knowledge/embeddings/base.py` — `Embedder` ABC (the interface: `model_id`,
  `dimension`, `embed_documents`, `embed_query`) plus the shared `l2_normalize`
  helper that implementations use to honour the "cosine == dot" contract.
- `src/knowledge/embeddings/openai_compat.py` — `OpenAICompatibleEmbedder`,
  a client for any OpenAI-compatible embeddings API (`GET /v1/models`,
  `POST /v1/embeddings`). Construction is side-effect free;
  `model_id`/`dimension` are resolved lazily on first access.
- `src/knowledge/config.py` — the **composition root**: `EmbeddingConfig`
  (pydantic) selects the implementation via `build_embedder()`.

Adding a new embedding backend means: implement `Embedder`, then register it
in `build_embedder()`. Never let concrete providers leak into core modules.

## Conventions

- Every module starts with `from __future__ import annotations`.
- Full type annotations everywhere; use PEP 604 unions (`str | None`), not
  `Optional`.
- Module-level docstrings explaining the "why" (see `openai_compat.py` for the style).
- Retry/backoff logic follows the pattern in `openai_compat.py` (`RETRYABLE_STATUS`,
  exponential backoff) — reuse it rather than re-implementing.

## Code Quality

Modern, maintainable Python (3.13):

- **Modern typing**: use `TypedDict` for structured dict payloads, `Self` for
  self-returning methods, `TypeAlias` for complex types; prefer
  `@dataclass(slots=True)` / pydantic over hand-written `__init__`.
- **Exception chains**: preserve context with `raise ... from exc` when
  re-raising — never discard the original exception.
- **No magic numbers**: name constants (see the retry backoff in `openai_compat.py`).
- **Dependencies**: adding a new dependency requires a reason; prefer the
  standard library first.
- **Tests**: assert on exception details where relevant
  (`pytest.raises(..., match=...)`), not just the exception type.
- **Tooling escapes**: when a linter/type-checker can't be satisfied for
  uncontrollable reasons (e.g. incomplete third-party stubs), prefer the
  narrowest escape: inline `# pyright: ignore[...]` / `# noqa` with a comment
  explaining why, over file-level or global downgrades. Keep a known-exemptions
  list here and revisit it after dependency upgrades.

## Commits & Attribution

- When AI assistance is used for non-trivial work (e.g. writing code), add an
  `Assisted-by` trailer to the commit message, formatted as
  `<tool> + <model>` (tool only if one was used), for example:
  `Assisted-by: Copilot Chat + DeepSeek V4 Flash`.
- Never use `Co-authored-by` to credit AI contributions.

## Testing

- Tests are **hermetic**: a fake OpenAI-compatible server is built on
  `httpx.MockTransport` (see `tests/test_openai_compat.py`). Never require a real
  server or network.
- Fake servers record requests so tests can assert on payloads
  (e.g. `normalize` flag, batching sizes).
- Keep new tests in the same style: no fixtures on the network, assert both
  behavior and requests sent.
