"""Unit tests for `lode.lexical.preview.tokenize_text`.

The preview must reproduce what each strategy's tokenizer actually stores at
index time — verified against the same FTS5 machinery, not a Python mirror.
The native ``simple``/``jieba`` strategies load their bundled extension, so
these tests stay hermetic (no network, no external services).
"""

from __future__ import annotations

import pytest

from lode.lexical import STRATEGIES, tokenize_text


def test_unicode61_splits_words_and_folds_case() -> None:
    tokens = tokenize_text(STRATEGIES["unicode61"], "Knowledge Mining engine")
    assert tokens == ["knowledge", "mining", "engine"]


def test_trigram_produces_three_char_grams() -> None:
    tokens = tokenize_text(STRATEGIES["trigram"], "abcd")
    assert tokens == ["abc", "bcd"]


def test_simple_indexes_han_characters_and_pinyin() -> None:
    tokens = tokenize_text(STRATEGIES["simple"], "知识")
    # Each Han character is indexed alongside its pinyin reading.
    assert tokens == ["z", "zhi", "知", "s", "shi", "z", "zhi", "识"]


def test_jieba_index_side_is_character_level() -> None:
    tokens = tokenize_text(STRATEGIES["jieba"], "知识挖掘")
    # At index time jieba stores the same Han-character + pinyin stream as
    # ``simple``; word-level segmentation happens on the query side via
    # ``jieba_query``, so whole-word terms never appear in the stored stream.
    assert "知" in tokens
    assert "挖" in tokens
    assert "知识" not in tokens


def test_empty_text_yields_no_tokens() -> None:
    for name in ("unicode61", "trigram"):
        assert tokenize_text(STRATEGIES[name], "") == []


def test_duplicates_are_kept_in_document_order() -> None:
    tokens = tokenize_text(STRATEGIES["unicode61"], "dog chases dog")
    assert tokens == ["dog", "chases", "dog"]


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_returns_a_list(name: str) -> None:
    result = tokenize_text(STRATEGIES[name], "mixed 中文 text")
    assert isinstance(result, list)
    assert all(isinstance(token, str) for token in result)
