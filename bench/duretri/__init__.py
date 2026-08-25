"""DuRetrieval tokenizer benchmark.

Compares FTS5 tokenizers (unicode61 / trigram / simple) on the MTEB
DuRetrieval Chinese retrieval task. Tokenizer and query strategy are treated
as two independent dimensions so we can measure both the recommended pairing
and cross combinations.
"""
