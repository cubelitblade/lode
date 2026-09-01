# Lode

[![CI](https://github.com/cubelitblade/lode/actions/workflows/ci.yml/badge.svg)](https://github.com/cubelitblade/lode/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lode-cli.svg)](https://pypi.org/project/lode-cli/)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
[![License](https://img.shields.io/github/license/cubelitblade/lode)](https://github.com/cubelitblade/lode/blob/main/LICENSE)

> No need to build another knowledge base. Your workspace is already a lode of knowledge.

Lode is a local-first knowledge mining engine that turns your workspace into a searchable knowledge lode.

> [!IMPORTANT]
> Lode is currently in pre-release stage.
> Features may be incomplete, behavior may change, and breaking changes may occur before the first stable release.

## Why Lode?

Your knowledge already exists.

It lives in:

- Documentation explaining concepts and decisions.
- Markdown notes capturing project knowledge.
- Design documents describing systems and workflows.
- Text files accumulated throughout development.

The problem is not storing more knowledge.

The problem is finding the right piece of knowledge when you need it.

Lode helps AI agents access workspace-specific knowledge by indexing existing documents and exposing them through semantic and lexical retrieval.

Instead of relying only on general-purpose models, agents can access the information that belongs to your projects.

## What Lode is not?

### Not a complete RAG application

Lode does not try to handle the entire RAG workflow.

Generation and reasoning remains the responsibility of your AI agents. Lode focuses on the retrieval layer:
finding relevant knowledge and providing useful context.

### Not another knowledge base

Lode does not require you to build and maintain a separate knowledge repository.

Knowledge is often scattered across documentation, notes, source code, and other project artifacts.
Lode helps discover existing knowledge instead of asking you to manually collect it.

### Not a replacement for your workspace

Lode does not ask you to reorganize your files or move your data into another system.

Your workspace remains the source of truth. Lode builds a retrieval layer on top of it without changing how you work.

## Installation

### From PyPI

Install with `uv`

```bash
uv tool install lode-cli
```

or with `pip`

```bash
pip install lode-cli
```

### From source

1. Clone the repository:

    ```bash
    git clone https://github.com/cubelitblade/lode.git
    cd lode
    ```

2. Install:

    ```bash
    uv tool install .
    ```

3. Verify:

    ```bash
    lode --help
    ```

## Quickstart

### 1. Provide an embedding provider

Lode does not ship with an embedding model. Bring your own embedding provider.

Lode supports:
- OpenAI-compatible endpoint
- Hugging Face Text Embeddings Inference (TEI) native endpoint
- Ollama endpoint

> [!TIP]
> The easiest way to start is using Hugging Face TEI:
>
> For example:
>
> ```bash
> mkdir -p $PWD/data
>
> model=Qwen/Qwen3-Embedding-0.6B
> volume=$PWD/data
>
> docker run --gpus all \
>   -p 8080:80 \
>   -v $volume:/data \
>   --pull always \
>   ghcr.io/huggingface/text-embeddings-inference:cuda-latest \
>   --model-id $model
> ```
>
> For more information, see [Hugging Face: Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/index).

### 2. Configure Lode

Set your embedding provider:

```bash
lode config set embedding.provider "openai_compatible" --scope user
lode config set embedding.openai_compatible.endpoint <endpoint> --scope user
lode config set embedding.model <model-name> --scope user
lode config set embedding.openai_compatible.key <api-key> --scope user # optional
```

For TEI native or Ollama endpoints, use `tei_native` or `ollama` instead.

Workspace-specific configuration can also be created using workspace scope.

### 3. Survey your workspace

Discover documents in your workspace:

```bash
lode survey
```

Survey builds the map of your workspace knowledge.

### 4. Mine your workspace

Generate embeddings and store indexed chunks:

```bash
lode mine
```

Mining processes discovered documents, generates embeddings, and stores searchable indexes locally in SQLite.

### 5. Prospect knowledge

Search your workspace:

```bash
lode prospect <query>
```

Lode performs hybrid retrieval using semantic similarity and BM25 lexical matching.

It combines meaning-based search with exact keyword matching, then returns candidate chunks with scores and digests.

### 6. Dig a chunk

Retrieve a specific chunk:

```bash
lode dig <digest>
```

This returns the complete content associated with the digest.

### 7. Assay a chunk

To understand how a chunk was scored, or why it was included or excluded from candidates, run:

```bash
lode assay why <digest>
```

This provides a detailed report explaining how this chunk was evaluated.

## FAQ

### Why does Lode fail to load SQLite extensions on macOS?

#### Possible reason

Lode requires SQLite extension loading for certain features.

Some Python builds, especially those linked against SQLite libraries without
loadable extension support, may not provide
`sqlite3.Connection.enable_load_extension()`.

This depends on how Python and SQLite were built, not only on the Python version.
CPython documents that loadable SQLite extension support is disabled by default
and notes macOS as a notable platform where the underlying SQLite library may
lack this capability.

You can verify the capability with:

```python
import sqlite3

print(hasattr(sqlite3.Connection, "enable_load_extension"))

```

#### Solution

- Use a separately installed Python distribution instead of the system-provided Python.
- In CI environments, explicitly configure the Python version and interpreter used by `uv`
  to avoid relying on the runner's preinstalled Python.

> For more information, see:
> - https://github.com/python/cpython/blob/main/Doc/library/sqlite3.rst
> - https://github.com/python/cpython/blob/main/Doc/using/configure.rst
