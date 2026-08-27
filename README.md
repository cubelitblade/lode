# Lode

> No need to build another knowledge base. Your workspace is already a lode of knowledge.

Lode is a local-first knowledge mining engine that turns your workspace into a searchable knowledge lode.

> [!WARNING]
> Lode is currently in early development.
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

## Install

### Install from source

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

### 1. Provide an embedding server

Lode does not ship with an embedding model. Bring your own embedding provider.
Lode supports local embedding servers such as Text Embeddings Inference (TEI), as well as OpenAI-compatible embedding APIs.

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
lode config set embedding.api.type "openai_compatible" --scope user
lode config set embedding.api.endpoint <endpoint> --scope user
lode config set embedding.model <model-name> --scope user
lode config set embedding.api.key <api-key> --scope user # optional
```
Workspace-specific configuration can also be created using workspace scope.

### 3. Survey a workspace

Discover documents in your workspace:

```bash
lode --workspace <path> survey
```

Survey builds the map of your knowledge lode.

### 4. Mine a workspace

Generate embeddings and store indexed chunks:

```bash
lode --workspace <path> mine
```

Mining processes discovered documents, generates embeddings, and stores searchable indexes locally in SQLite.

### 5. Prospect knowledge

Search your workspace:

```bash
lode --workspace <path> prospect <query>
```

Lode performs hybrid retrieval using semantic similarity and BM25 lexical matching.

It combines meaning-based search with exact keyword matching, then returns candidate chunks with scores and digests.

### 6. Dig the ore

Retrieve a specific chunk:

```bash
lode --workspace <path> dig <digest>
```

This returns the complete content associated with the digest.
