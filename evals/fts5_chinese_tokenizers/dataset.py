"""Load the DuRetrieval dataset (corpus, queries, qrels).

The dataset is fetched through ``datasets`` and cached by HuggingFace; nothing
is persisted by this evaluation beyond that cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from datasets import load_dataset  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

REPO = "mteb/DuRetrieval"


@dataclass(frozen=True, slots=True)
class DuRetrieval:
    """The three pieces of the retrieval task, keyed by document id."""

    # (doc_id, text) pairs.
    corpus: list[tuple[str, str]]
    # (query_id, text) pairs.
    queries: list[tuple[str, str]]
    # query_id -> set of relevant doc_ids.
    qrels: dict[str, set[str]]


def _rows(config: str) -> list[dict[str, Any]]:
    """Load one config's dev split as a list of row dicts.

    ``datasets`` ships no type stubs, so the returned rows are cast to a plain
    dict shape to keep the rest of the module fully typed.
    """
    ds = load_dataset(REPO, config)["dev"]  # pyright: ignore[reportUnknownVariableType]
    return cast(list[dict[str, Any]], list(ds))


def load_duretri() -> DuRetrieval:
    """Load the dev split of DuRetrieval."""
    corpus = [(row["_id"], row["text"]) for row in _rows("corpus")]
    queries = [(row["_id"], row["text"]) for row in _rows("queries")]
    qrels: dict[str, set[str]] = {}
    for row in _rows("default"):
        qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return DuRetrieval(corpus=corpus, queries=queries, qrels=qrels)
