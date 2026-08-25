"""Table-driven runner for the DuRetrieval tokenizer benchmark.

Builds an FTS5 index for each tokenizer on a temporary on-disk database (so we
can measure index size), runs every (tokenizer, query strategy) combination,
and reports Recall@k / Precision@k / MRR@k plus index size.

Usage:
    uv run python -m bench.duretri.run [--max-queries N] [--top-k K]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from bench.duretri.data import load_duretri
from bench.duretri.metrics import mrr_at_k, precision_at_k, recall_at_k
from bench.duretri.query import QUERY_STRATEGIES, RECOMMENDED_STRATEGY, SIMPLE_HELPER_SQL, match
from bench.duretri.tokenizers import TOKENIZERS, create_table

# Default number of labeled queries to evaluate (full dev set is 2000).
DEFAULT_MAX_QUERIES = int(os.environ.get("MAX_QUERIES", "200"))
DEFAULT_TOP_K = 10


def build_index(tokenizer_name: str, corpus: list[tuple[str, str]]) -> tuple[sqlite3.Connection, int, Path]:
    """Build an FTS5 index for ``corpus`` on a temp file.

    Returns ``(conn, size_bytes, tmpdir)``. The database lives in a temporary
    directory so we can measure the on-disk index size; the caller is
    responsible for closing the connection and deleting ``tmpdir``.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="lode-bench-"))
    db_path = tmpdir / "index.db"
    conn = sqlite3.connect(db_path)
    create_table(conn, tokenizer_name)
    rows = ((i, text) for i, (_, text) in enumerate(corpus, start=1))
    conn.executemany("INSERT INTO t(rowid, text) VALUES (?, ?)", rows)
    conn.commit()
    size = db_path.stat().st_size
    return conn, size, tmpdir


def search(
    conn: sqlite3.Connection,
    strategy_name: str,
    query: str,
    k: int,
) -> list[int]:
    """Return the top ``k`` rowids for ``query`` under the given strategy."""
    match_arg = match(conn, strategy_name, query)
    try:
        if QUERY_STRATEGIES[strategy_name].uses_helper:
            sql = f"SELECT rowid FROM t WHERE t MATCH {SIMPLE_HELPER_SQL[strategy_name]} ORDER BY bm25(t) LIMIT ?"
            rows = conn.execute(sql, (match_arg, k)).fetchall()
        else:
            rows = conn.execute(
                "SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?",
                (match_arg, k),
            ).fetchall()
        return [int(r[0]) for r in rows]
    except sqlite3.OperationalError:
        return []


def evaluate(
    conn: sqlite3.Connection,
    strategy_name: str,
    labeled: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    corpus_id_to_rowid: dict[str, int],
    k: int,
) -> dict[str, float]:
    """Average Recall@k / Precision@k / MRR@k over the labeled queries."""
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    for qid, qtext in labeled:
        relevant = {corpus_id_to_rowid[cid] for cid in qrels[qid] if cid in corpus_id_to_rowid}
        retrieved = search(conn, strategy_name, qtext, k)
        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        mrrs.append(mrr_at_k(retrieved, relevant, k))
    n = len(labeled)
    return {
        "recall@k": sum(recalls) / n if n else 0.0,
        "precision@k": sum(precisions) / n if n else 0.0,
        "mrr@k": sum(mrrs) / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DuRetrieval tokenizer benchmark")
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    data = load_duretri()
    corpus_id_to_rowid = {cid: i + 1 for i, (cid, _) in enumerate(data.corpus)}
    labeled = [(qid, qtext) for qid, qtext in data.queries if qid in data.qrels][: args.max_queries]
    print(f"corpus={len(data.corpus)} queries={len(data.queries)} labeled={len(labeled)} top_k={args.top_k}")

    # Build each tokenizer's index once and record its size.
    indexes: dict[str, tuple[sqlite3.Connection, int]] = {}
    tmpdirs: list[Path] = []
    try:
        for name in TOKENIZERS:
            conn, size, tmpdir = build_index(name, data.corpus)
            indexes[name] = (conn, size)
            tmpdirs.append(tmpdir)
            print(f"index[{name}] size={size} bytes")

        # Table-driven: every (tokenizer, strategy) combination under test.
        # Recommended pairings are marked; cross combinations are kept to
        # measure the indexing/query interaction.
        combinations: list[tuple[str, str, bool]] = []
        for tokenizer_name in TOKENIZERS:
            for strategy_name in QUERY_STRATEGIES:
                recommended = RECOMMENDED_STRATEGY.get(tokenizer_name) == strategy_name
                combinations.append((tokenizer_name, strategy_name, recommended))

        print(f"\n{'tokenizer':<10} {'strategy':<10} {'rec@k':<8} {'prec@k':<8} {'mrr@k':<8} {'rec?':<5}")
        print("-" * 55)
        for tokenizer_name, strategy_name, recommended in combinations:
            conn, _ = indexes[tokenizer_name]
            metrics = evaluate(conn, strategy_name, labeled, data.qrels, corpus_id_to_rowid, args.top_k)
            mark = "yes" if recommended else ""
            print(
                f"{tokenizer_name:<10} {strategy_name:<10} "
                f"{metrics['recall@k']:<8.4f} {metrics['precision@k']:<8.4f} {metrics['mrr@k']:<8.4f} {mark:<5}"
            )
    finally:
        for conn, _ in indexes.values():
            conn.close()
        # Clean up every temporary database directory.
        import shutil

        for tmpdir in tmpdirs:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
