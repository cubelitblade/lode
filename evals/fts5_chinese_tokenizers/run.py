"""Runner for the FTS5 Chinese tokenizer evaluation over MTEB DuRetrieval.

Builds an FTS5 index for each index-side tokenizer on a temporary on-disk
database, runs each legal (index tokenizer, query strategy) pairing, and
prints two tables: a cost summary per pairing (on-disk index size, index
build time, evaluation wall time) and Recall@k / Precision@k / MRR@k plus
the zero-hit rate (share of queries returning an empty result set) bucketed
by query length.

Public sentence-level benchmarks contain almost no short queries (verified
across the MTEB Chinese retrieval sets), yet short terms are common in real
search and stress tokenizers at their weakest regimes: trigram cannot emit a
MATCH expression below three characters, and a three-character query gives
it only a single gram. The short buckets are therefore populated
by synthesized probe queries — contiguous substrings of labeled queries that
occur in at least one relevant document, reusing the parent's relevance
judgments — so a low score there reflects a genuine matching failure rather
than missing labels.

Usage:
    uv run python -m evals.fts5_chinese_tokenizers.run [--max-queries N] [--top-k K]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from evals.fts5_chinese_tokenizers.dataset import load_duretri
from evals.fts5_chinese_tokenizers.metrics import mrr_at_k, precision_at_k, recall_at_k
from lode.lexical import HELPER_SQL, STRATEGIES

# Index-side tokenizers under test. ``jieba`` is a query-side strategy that
# reuses the ``simple`` index, so it does not build an index of its own.
TOKENIZERS = {name: s for name, s in STRATEGIES.items() if name != "jieba"}

# Default number of labeled queries to evaluate (full dev set is 2000).
# Probes derive from the same queries, so this also bounds probe volume.
DEFAULT_MAX_QUERIES = int(os.environ.get("MAX_QUERIES", "200"))
DEFAULT_TOP_K = 10

# Character lengths of synthesized probe queries.
PROBE_LENGTHS = (1, 2, 3)

# Output order of length buckets; "all" aggregates every bucket. Boundaries
# follow tokenizer mechanics rather than digit grouping: 2 is trigram's
# hard floor, 3 its single-gram minimum.
BUCKET_ORDER = ("1", "2", "3", "4-9", "10+", "all")

# Legal (index tokenizer, query strategy) pairings. Helper-backed strategies
# (simple/jieba) can only run where the ``simple`` extension is loaded — i.e.
# against the ``simple`` index — and every other strategy pairs only with its
# own tokenizer; anything else is guaranteed-empty noise.
LEGAL_PAIRINGS: tuple[tuple[str, str], ...] = (
    ("unicode61", "unicode61"),
    ("trigram", "trigram"),
    ("simple", "simple"),
    ("simple", "jieba"),
)


def _is_han(text: str) -> bool:
    """Whether ``text`` consists solely of Han characters."""
    return all("\u4e00" <= ch <= "\u9fff" for ch in text)


def bucket_of(text: str) -> str:
    """Name of the query-length bucket ``text`` falls into."""
    n = len(text)
    if n <= 2:
        return str(n)
    if n == 3:
        return "3"
    if n <= 9:
        return "4-9"
    return "10+"


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One evaluation query with precomputed relevant rowids."""

    qid: str
    text: str
    bucket: str
    relevant: frozenset[int]


def build_probes(
    labeled: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    corpus_id_to_rowid: dict[str, int],
    corpus_text: dict[str, str],
) -> list[EvalQuery]:
    """Synthesize short probe queries from labeled queries.

    A probe is a contiguous Han-character substring of its parent query that
    occurs in at least one of the parent's relevant documents, so every
    strategy could in principle retrieve it — failing anyway exposes a
    structural inability to match that query length. Probes inherit the
    parent's relevance judgments and are deduplicated across parents
    (relevant sets unioned).
    """
    probes: dict[str, set[int]] = {}
    for qid, text in labeled:
        relevant_docs = [(cid, corpus_text[cid]) for cid in qrels.get(qid, set()) if cid in corpus_text]
        if not relevant_docs:
            continue
        for n in PROBE_LENGTHS:
            for i in range(len(text) - n + 1):
                sub = text[i : i + n]
                if not _is_han(sub):
                    continue
                rowids = {corpus_id_to_rowid[cid] for cid, doc in relevant_docs if sub in doc}
                if rowids:
                    probes.setdefault(sub, set()).update(rowids)
    return [
        EvalQuery(qid=f"probe:{sub}", text=sub, bucket=bucket_of(sub), relevant=frozenset(rowids))
        for sub, rowids in sorted(probes.items())
    ]


def build_eval_queries(
    labeled: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    corpus_id_to_rowid: dict[str, int],
    corpus_text: dict[str, str],
) -> list[EvalQuery]:
    """Natural labeled queries plus synthesized probes, ready for evaluation."""
    queries: list[EvalQuery] = []
    for qid, text in labeled:
        relevant = frozenset(corpus_id_to_rowid[cid] for cid in qrels.get(qid, set()) if cid in corpus_id_to_rowid)
        if relevant:
            queries.append(EvalQuery(qid=qid, text=text, bucket=bucket_of(text), relevant=relevant))
    queries.extend(build_probes(labeled, qrels, corpus_id_to_rowid, corpus_text))
    return queries


def _create_table(conn: sqlite3.Connection, tokenizer_name: str) -> None:
    """Create the FTS5 table ``t`` for ``tokenizer_name`` on ``conn``.

    Mirrors how the schema layer owns DDL in the library: the strategy
    contributes its setup and ``tokenize=`` clause, the statement stays here.
    """
    strategy = STRATEGIES[tokenizer_name]
    strategy.setup(conn)
    conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(text, tokenize='{strategy.tokenize_clause}')")


def build_index(tokenizer_name: str, corpus: list[tuple[str, str]]) -> tuple[sqlite3.Connection, int, Path]:
    """Build an FTS5 index for ``corpus`` on a temp file.

    Returns ``(conn, size_bytes, tmpdir)``. The database lives in a temporary
    directory so we can measure the on-disk index size; the caller is
    responsible for closing the connection and deleting ``tmpdir``.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="lode-eval-"))
    db_path = tmpdir / "index.db"
    conn = sqlite3.connect(db_path)
    _create_table(conn, tokenizer_name)
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
    strategy = STRATEGIES[strategy_name]
    try:
        if strategy.uses_helper:
            sql = f"SELECT rowid FROM t WHERE t MATCH {HELPER_SQL[strategy_name]} ORDER BY bm25(t) LIMIT ?"
            rows = conn.execute(sql, (query, k)).fetchall()
        else:
            rows = conn.execute(
                "SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?",
                (strategy.query(query), k),
            ).fetchall()
        return [int(r[0]) for r in rows]
    except sqlite3.OperationalError:
        # Malformed MATCH expressions (e.g. trigram on < 3 characters) count
        # as zero hits rather than aborting the run.
        return []


@dataclass(slots=True)
class BucketStats:
    """Aggregated per-bucket metrics for one (tokenizer, strategy) pair."""

    n: int = 0
    recall_sum: float = 0.0
    precision_sum: float = 0.0
    mrr_sum: float = 0.0
    zero_hits: int = 0

    def add(self, retrieved: list[int], relevant: frozenset[int], k: int) -> None:
        self.n += 1
        self.recall_sum += recall_at_k(retrieved, relevant, k)
        self.precision_sum += precision_at_k(retrieved, relevant, k)
        self.mrr_sum += mrr_at_k(retrieved, relevant, k)
        if not retrieved:
            self.zero_hits += 1

    def averages(self) -> tuple[float, float, float, float]:
        """(recall@k, precision@k, mrr@k, zero-hit rate) over this bucket."""
        if not self.n:
            return 0.0, 0.0, 0.0, 0.0
        return (
            self.recall_sum / self.n,
            self.precision_sum / self.n,
            self.mrr_sum / self.n,
            self.zero_hits / self.n,
        )


def evaluate(
    conn: sqlite3.Connection,
    strategy_name: str,
    queries: list[EvalQuery],
    k: int,
) -> dict[str, BucketStats]:
    """Per-bucket metrics for one (tokenizer, strategy) combination."""
    stats = {name: BucketStats() for name in BUCKET_ORDER}
    for q in queries:
        retrieved = search(conn, strategy_name, q.text, k)
        stats[q.bucket].add(retrieved, q.relevant, k)
        stats["all"].add(retrieved, q.relevant, k)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FTS5 Chinese tokenizer evaluation on DuRetrieval: legal "
            "(tokenizer, query strategy) pairings scored per query-length "
            "bucket, with short buckets filled by synthesized substring probes"
        ),
    )
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    data = load_duretri()
    corpus_id_to_rowid = {cid: i + 1 for i, (cid, _) in enumerate(data.corpus)}
    corpus_text = dict(data.corpus)
    labeled = [(qid, qtext) for qid, qtext in data.queries if qid in data.qrels][: args.max_queries]
    queries = build_eval_queries(labeled, data.qrels, corpus_id_to_rowid, corpus_text)
    probes = sum(1 for q in queries if q.qid.startswith("probe:"))
    print(
        f"corpus={len(data.corpus)} labeled={len(labeled)} probes={probes} "
        f"eval_queries={len(queries)} top_k={args.top_k}"
    )
    print("note: buckets 1-3 include synthesized substrings of labeled queries that occur in a relevant document")

    # Build each tokenizer's index once, recording size and build time.
    indexes: dict[str, sqlite3.Connection] = {}
    index_costs: dict[str, tuple[int, float]] = {}  # tokenizer -> (bytes, build_secs)
    tmpdirs: list[Path] = []
    try:
        for name in TOKENIZERS:
            started = time.perf_counter()
            conn, size, tmpdir = build_index(name, data.corpus)
            indexes[name] = conn
            index_costs[name] = (size, time.perf_counter() - started)
            tmpdirs.append(tmpdir)

        # Evaluate every legal pairing, keeping its wall time for the summary.
        evaluations: list[tuple[str, str, dict[str, BucketStats], float]] = []
        for tokenizer_name, strategy_name in LEGAL_PAIRINGS:
            started = time.perf_counter()
            stats = evaluate(indexes[tokenizer_name], strategy_name, queries, args.top_k)
            evaluations.append((tokenizer_name, strategy_name, stats, time.perf_counter() - started))

        # Cost summary: one row per pairing; pairings sharing an index repeat
        # that index's size and build time.
        header = f"{'tokenizer':<10} {'strategy':<10} {'index_bytes':>12} {'build_secs':>11} {'eval_secs':>10}"
        print(f"\n{header}")
        print("-" * len(header))
        for tokenizer_name, strategy_name, _, eval_secs in evaluations:
            size, build_secs = index_costs[tokenizer_name]
            print(f"{tokenizer_name:<10} {strategy_name:<10} {size:>12} {build_secs:>11.1f} {eval_secs:>10.2f}")

        # Quality metrics per query-length bucket.
        header = (
            f"{'tokenizer':<10} {'strategy':<10} {'bucket':<7} {'n':>6} "
            f"{'rec@k':<8} {'prec@k':<8} {'mrr@k':<8} {'zero%':<7}"
        )
        print(f"\n{header}")
        print("-" * len(header))
        for tokenizer_name, strategy_name, stats, _ in evaluations:
            for bucket in BUCKET_ORDER:
                rec, prec, mrr, zero = stats[bucket].averages()
                print(
                    f"{tokenizer_name:<10} {strategy_name:<10} {bucket:<7} {stats[bucket].n:>6} "
                    f"{rec:<8.4f} {prec:<8.4f} {mrr:<8.4f} {zero:<7.2%}"
                )
    finally:
        for conn in indexes.values():
            conn.close()
        # Clean up every temporary database directory.
        for tmpdir in tmpdirs:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
