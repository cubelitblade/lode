"""FTS5 Chinese tokenizer evaluation on MTEB DuRetrieval.

Measures how well each FTS5 lexical setup retrieves Chinese passages: the
built-in unicode61 / trigram tokenizers plus the bundled native ``simple``
tokenizer with its ``simple_query`` / ``jieba_query`` helpers. Output is two
tables: a cost summary per legal index/query pairing (on-disk index size,
build and evaluation wall time) and per-query-length-bucket recall /
precision / MRR with a zero-hit rate. The 1-3 character buckets are populated or reinforced by synthesized substring
probes because public sentence-level benchmarks contain almost no such queries.
"""
