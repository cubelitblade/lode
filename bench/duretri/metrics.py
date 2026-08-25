"""Retrieval metrics: Recall@k, Precision@k, and MRR@k.

All metrics are computed from a ranked list of retrieved rowids and the set of
relevant rowids for a query. ``k`` bounds how many of the top results count.
"""

from __future__ import annotations


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant documents found in the top ``k`` results."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / k


def mrr_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Reciprocal rank of the first relevant result within the top ``k``.

    Returns 0.0 when no relevant document appears in the top ``k``.
    """
    for rank, rowid in enumerate(retrieved[:k], start=1):
        if rowid in relevant:
            return 1.0 / rank
    return 0.0
