"""Tests for document extractor evaluation metrics."""

from __future__ import annotations

import pytest
from evals.document_extractors.metrics import (
    anchor_order_accuracy,
    content_view,
    invalid_unicode_character_count,
    markdown_content_projection,
    markdown_delimiters_balanced,
    markdown_heading_accuracy,
    markdown_heading_sequence,
    markdown_plain_text,
    markdown_table_count,
    ngram_prf,
    normalized_edit_similarity,
    segment_boundary_accuracy,
    segment_content_boundary_accuracy,
    segment_provenance_accuracy,
    strict_view,
    table_row_accuracy,
)


def test_normalization_views_preserve_or_collapse_structure() -> None:
    source = "Cafe\u0301  \r\n下一行 \t"
    assert strict_view(source) == "Café\n下一行"
    assert content_view(source) == "Café 下一行"


def test_ngram_prf_is_exact_for_equivalent_whitespace() -> None:
    assert ngram_prf("中文\n内容", "中文  内容") == (1.0, 1.0, 1.0)


def test_ngram_prf_reports_missing_content() -> None:
    precision, recall, f1 = ngram_prf("abcdef", "abcdefghi")
    assert precision == 1.0
    assert recall < 1.0
    assert precision > f1 > recall


def test_normalized_edit_similarity() -> None:
    assert normalized_edit_similarity("abc", "abc") == 1.0
    assert normalized_edit_similarity("abc", "axc") == pytest.approx(2 / 3)
    assert normalized_edit_similarity("", "abc") == 0.0
    assert normalized_edit_similarity("kitten", "sitting") == pytest.approx(4 / 7)


def test_normalized_edit_similarity_handles_long_inputs_exactly() -> None:
    actual = "a" * 20_000 + "x"
    expected = "a" * 20_000 + "y"
    assert normalized_edit_similarity(actual, expected) == pytest.approx(20_000 / 20_001)


def test_anchor_order_accuracy_requires_order() -> None:
    anchors = ("first", "second", "third")
    assert anchor_order_accuracy("first second third", anchors) == 1.0
    assert anchor_order_accuracy("second first third", anchors) == pytest.approx(2 / 3)


def test_table_row_accuracy_requires_exact_pipe_delimited_rows_in_order() -> None:
    expected = ("样本 | 数值", "A | 1.0")
    assert table_row_accuracy("前言\n样本 | 数值\nA | 1.0\n结尾", expected) == 1.0
    assert table_row_accuracy("样本\t数值\nA\t1.0", expected) == 0.0
    assert table_row_accuracy("A | 1.0\n样本 | 数值", expected) == pytest.approx(0.5)


def test_markdown_delimiter_balance_is_only_a_parity_check() -> None:
    assert markdown_delimiters_balanced("**strong** and `code`")
    assert markdown_delimiters_balanced("```text\nvalue\n```")
    assert not markdown_delimiters_balanced("**truncated")
    assert not markdown_delimiters_balanced("```text\ntruncated")


def test_markdown_table_count_uses_gfm_separator_rows() -> None:
    markdown = "| Name | Value |\n| --- | --- |\n| A | 1 |\n\nText"
    assert markdown_table_count(markdown) == 1
    assert markdown_table_count("| not a table |\n| value |") == 0


def test_markdown_plain_text_only_removes_block_syntax() -> None:
    source = "# Heading\n\n- item with `literal` and * stars *\n```text\ncode\n```"
    assert markdown_plain_text(source) == "Heading\n\nitem with `literal` and * stars *\ncode"


def test_markdown_content_projection_removes_generated_structure() -> None:
    source = "# **Heading**\n\n| Name | Value |\n| --- | --- |\n| A | [one](https://example.test) |"
    assert markdown_content_projection(source) == "Heading\n\nName | Value\nA | one"


def test_markdown_heading_sequence_uses_readable_titles() -> None:
    assert markdown_heading_sequence("# **Title**\n### [Section](#section)") == (
        (1, "Title"),
        (3, "Section"),
    )


def test_markdown_heading_accuracy_checks_level_and_sequence() -> None:
    expected = ((1, "Title"), (2, "Section"))
    assert markdown_heading_accuracy("# Title\n## Section", expected) == 1.0
    assert markdown_heading_accuracy("# Title\n### Section", expected) == pytest.approx(0.5)


def test_invalid_unicode_character_count_includes_replacement_and_noncharacters() -> None:
    assert invalid_unicode_character_count("ok\ufffd\uffff\U0010fffe") == 3
    assert invalid_unicode_character_count("中文 text") == 0


def test_segment_metrics_penalize_wrong_boundaries_and_provenance() -> None:
    expected = (("Heading\nBody", "Heading", None), ("Next\nText", "Heading / Next", None))
    assert segment_boundary_accuracy(expected, expected) == 1.0
    assert segment_provenance_accuracy(expected, expected) == 1.0

    merged = (("Heading\nBody\nNext\nText", "", None),)
    assert segment_boundary_accuracy(merged, expected) == 0.0
    assert segment_provenance_accuracy(merged, expected) == 0.0

    wrong_heading = (("Heading\nBody", "", None), ("Next\nText", "", None))
    assert segment_boundary_accuracy(wrong_heading, expected) == 1.0
    assert segment_provenance_accuracy(wrong_heading, expected) == 0.0


def test_pdf_boundary_metric_ignores_only_internal_whitespace() -> None:
    expected = (("First paragraph\nSecond paragraph", "", 1),)
    actual = (("First paragraph\n\nSecond paragraph", "", 1),)
    assert segment_boundary_accuracy(actual, expected) == 0.0
    assert segment_content_boundary_accuracy(actual, expected) == 1.0

    split = (("First paragraph", "", 1), ("Second paragraph", "", 1))
    assert segment_content_boundary_accuracy(split, expected) == 0.0
