# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Context

`lode` is a local-first knowledge mining engine that turns workspace documents into searchable knowledge.

The goal is to make existing project knowledge accessible to both humans and AI agents through semantic search and MCP integration.

Lode focuses on document-oriented knowledge retrieval. Do not assume that Lode understands source code semantics or provides code intelligence features.

Current status:

* Early development stage.
* Embedding layer is implemented.
* Document loading, indexing, retrieval, and MCP integration are under development.

The product vision and roadmap live in `.vscode/docs/PLAN.md`. Read it before making large changes, and link to it instead of duplicating roadmap content.

## Engineering Principles

### Dependency inversion

Business logic depends on abstractions, not concrete implementations.

For example:

* Embedding providers implement the `Embedder` interface.
* Core modules should not depend on specific providers such as OpenAI-compatible APIs.

When adding a new implementation:

1. Implement the required interface.
2. Keep provider-specific logic isolated.
3. Register it through the composition root.

### Composition root

Runtime dependencies are assembled in the configuration layer.

Current composition root:

* `src/lode/config.py`

Configuration models:

* `EmbeddingConfig`
* `RetrievalConfig`
* `IgnoreConfig`

Settings are loaded through `load_settings()`.

Configuration precedence:

```
defaults < TOML < environment variables < constructor kwargs
```

The TOML configuration source should remain replaceable for future formats.

### Scope control

Keep changes focused on the requested task.

* Do not add features without a concrete requirement.
* Do not introduce abstractions without a clear use case.
* Prefer incremental changes over large refactors.
* Do not assume future requirements.
* Avoid solving problems that are outside the current project scope.

## Architecture

Important modules:

### Embeddings

`src/lode/embeddings/`

Contains embedding interfaces and implementations.

* `base.py`

  * Defines the `Embedder` abstraction.
  * Provides:

    * `model_id`
    * `dimension`
    * `embed_documents`
    * `embed_query`
  * Contains shared `l2_normalize` logic.
  * The normalization contract ensures cosine similarity can be represented as dot product similarity.

* `openai_compat.py`

  * Implements `OpenAICompatibleEmbedder`.
  * Supports OpenAI-compatible embedding APIs:

    * `GET /v1/models`
    * `POST /v1/embeddings`
  * Construction should remain side-effect free.
  * Provider metadata should be resolved lazily when possible.

Adding a new embedding backend should not require changes to core business logic.

## Development Workflow

The project uses `uv`.

Requirements:

* Python >= 3.13
* `src/` layout

Common commands:

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyright
```

## Code Style

Follow modern Python practices.

### General rules

* Every module starts with:

  ```python
  from __future__ import annotations
  ```

* Use full type annotations.

* Prefer PEP 604 unions:

  ```python
  str | None
  ```

  instead of:

  ```python
  Optional[str]
  ```

### Modern typing

Prefer:

* `TypedDict` for structured dictionary payloads.
* `Self` for self-returning methods.
* `TypeAlias` for complex reusable types.
* `dataclass(slots=True)` or pydantic models over manually written constructors.

### Exceptions

Preserve exception context.

Use:

```python
raise NewError(...) from exc
```

when wrapping exceptions.

Never silently discard the original exception.

### Constants

Avoid magic numbers.

Use named constants for values with meaning.

### Dependencies

Before adding a dependency:

* Check whether the standard library is sufficient.
* Ensure the dependency provides meaningful value.
* Keep dependency additions justified.

### Tooling escapes

When linters or type checkers cannot be satisfied due to external limitations:

Prefer:

```python
# pyright: ignore[...]
```

or:

```python
# noqa
```

with a comment explaining why.

Avoid:

* File-wide ignores.
* Global configuration downgrades.

Keep known exceptions documented and revisit them after dependency upgrades.

### Comments

Comments should explain why, not what.

Add comments only for:

* Hidden constraints.
* Non-obvious invariants.
* Required workarounds.
* Intentionally surprising behavior.

If removing a comment would not confuse a competent maintainer, do not add it.

## Testing

Tests must remain hermetic.

Rules:

* Never require external services.
* Never depend on network availability.
* Use mock servers for HTTP APIs.

Current HTTP testing approach:

* Use `httpx.MockTransport` for fake OpenAI-compatible servers.
* Assert both returned behavior and outgoing requests.

When adding tests:

* Follow existing mock server patterns.
* Verify important request details when relevant.
* Test behavior, not implementation details.

## Commits & Pull Requests

Commit messages should describe the final outcome of the change.

Do not include implementation history or internal decision process unless it
is directly relevant to users or maintainers.

Avoid mentioning:

- rejected approaches;
- temporary implementations;
- removed experiments;
- internal debugging steps;
- implementation details that do not affect the final behavior.

Examples:

Good:

```text
feat: add support for custom base URLs
```

Bad:

```text
feat: try custom base URLs, remove hardcoded default, refactor client config
```

The commit history should explain what changed, not the complete story of how
the change was made.

### AI-assisted changes

When AI assistance is used for non-trivial code changes:

Add an `Assisted-by` trailer:

```text
Assisted-by: <tool> + <model>
```

Examples:

```text
Assisted-by: Copilot Chat + DeepSeek V4 Flash
```

Do not use `Co-authored-by` for AI contributions.
