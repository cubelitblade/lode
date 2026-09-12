"""Normalization and quality metrics for document extraction evaluations."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_MARKDOWN_BLOCK_PREFIX_RE = re.compile(r"^[ \t]*(?:#{1,6}|[-+*]|\d+[.)])[ \t]+")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^\|?(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MARKDOWN_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


def strict_view(text: str) -> str:
    """Normalize representation without hiding block or reading-order changes."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def content_view(text: str) -> str:
    """Normalize whitespace for cross-engine content comparisons."""
    return _WHITESPACE_RE.sub(" ", strict_view(text)).strip()


def ngram_prf(actual: str, expected: str, *, width: int = 3) -> tuple[float, float, float]:
    """Return multiset character n-gram precision, recall, and F1."""
    actual_grams = _ngrams(content_view(actual), width)
    expected_grams = _ngrams(content_view(expected), width)
    if not actual_grams and not expected_grams:
        return 1.0, 1.0, 1.0
    if not actual_grams or not expected_grams:
        return 0.0, 0.0, 0.0
    overlap = sum((actual_grams & expected_grams).values())
    precision = overlap / sum(actual_grams.values())
    recall = overlap / sum(expected_grams.values())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def normalized_edit_similarity(actual: str, expected: str) -> float:
    """Return one minus Levenshtein distance divided by the longer input."""
    left = strict_view(actual)
    right = strict_view(expected)
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    distance = _bit_parallel_levenshtein(left, right)
    return 1.0 - distance / max(len(left), len(right))


def anchor_order_accuracy(text: str, anchors: tuple[str, ...]) -> float:
    """Measure how much of an expected anchor sequence appears in order."""
    if not anchors:
        return 1.0
    haystack = content_view(text)
    cursor = 0
    found = 0
    for anchor in anchors:
        position = haystack.find(content_view(anchor), cursor)
        if position < 0:
            continue
        found += 1
        cursor = position + len(content_view(anchor))
    return found / len(anchors)


def table_row_accuracy(text: str, expected_rows: tuple[str, ...]) -> float:
    """Measure exact pipe-delimited table rows retained in reading order."""
    if not expected_rows:
        return 1.0
    lines = strict_view(text).splitlines()
    cursor = 0
    found = 0
    for expected_row in expected_rows:
        normalized_row = strict_view(expected_row)
        while cursor < len(lines):
            line = lines[cursor]
            cursor += 1
            if line == normalized_row:
                found += 1
                break
    return found / len(expected_rows)


def markdown_delimiters_balanced(markdown: str) -> bool:
    """Check delimiter parity relevant to the planned CLI completion strategy."""
    without_fences = markdown.replace("```", "")
    return all(
        count % 2 == 0
        for count in (
            markdown.count("```"),
            without_fences.count("`"),
            markdown.count("**"),
            markdown.count("__"),
            markdown.count("~~"),
        )
    )


def markdown_table_count(markdown: str) -> int:
    """Count GFM table blocks using their separator rows."""
    lines = strict_view(markdown).splitlines()
    return sum(
        index > 0
        and line.startswith("|")
        and bool(_MARKDOWN_TABLE_SEPARATOR_RE.match(line))
        and lines[index - 1].lstrip().startswith("|")
        for index, line in enumerate(lines)
    )


def markdown_plain_text(markdown: str) -> str:
    """Remove generated block syntax while retaining literal inline tokens."""
    lines: list[str] = []
    for line in strict_view(markdown).splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(_MARKDOWN_BLOCK_PREFIX_RE.sub("", line))
    return "\n".join(lines)


def markdown_content_projection(markdown: str) -> str:
    """Project canonical Markdown to readable content for quality metrics."""
    lines: list[str] = []
    for raw_line in strict_view(markdown).splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            continue
        if _MARKDOWN_TABLE_SEPARATOR_RE.match(line):
            continue
        line = _MARKDOWN_BLOCK_PREFIX_RE.sub("", line)
        if line.startswith("|") and line.endswith("|"):
            line = line[1:-1].strip()
        line = _MARKDOWN_LINK_RE.sub(r"\1", line)
        line = _MARKDOWN_TAG_RE.sub("", line)
        line = re.sub(r"([*_~`])", "", line)
        line = line.replace(r"\\", "\\")
        line = line.replace(r"\|", "|")
        lines.append(line)
    return "\n".join(lines)


def markdown_heading_sequence(markdown: str) -> tuple[tuple[int, str], ...]:
    """Return heading levels and readable titles in source order."""
    headings: list[tuple[int, str]] = []
    for line in strict_view(markdown).splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match is not None:
            headings.append((len(match.group(1)), markdown_content_projection(match.group(2))))
    return tuple(headings)


def markdown_heading_accuracy(markdown: str, expected: tuple[tuple[int, str], ...]) -> float:
    """Measure exact generated Markdown heading sequence and levels."""
    actual = tuple(
        (len(match.group(1)), match.group(2))
        for line in strict_view(markdown).splitlines()
        if (match := _MARKDOWN_HEADING_RE.match(line)) is not None
    )
    if not actual and not expected:
        return 1.0
    denominator = max(len(actual), len(expected))
    if denominator == 0:
        return 0.0
    matches = sum(left == right for left, right in zip(actual, expected, strict=False))
    return matches / denominator


def invalid_unicode_character_count(text: str) -> int:
    """Count replacement characters and Unicode noncharacters unsafe for display."""
    return sum(
        character == "\ufffd" or 0xFDD0 <= ord(character) <= 0xFDEF or ord(character) & 0xFFFF in {0xFFFE, 0xFFFF}
        for character in text
    )


def segment_boundary_accuracy(
    actual: tuple[tuple[str, str, int | None], ...],
    expected: tuple[tuple[str, str, int | None], ...],
) -> float:
    """Measure exact text boundaries while ignoring provenance fields."""
    if not actual and not expected:
        return 1.0
    denominator = max(len(actual), len(expected))
    if denominator == 0:
        return 0.0
    matches = sum(
        strict_view(actual_segment[0]) == strict_view(expected_segment[0])
        for actual_segment, expected_segment in zip(actual, expected, strict=False)
    )
    return matches / denominator


def segment_content_boundary_accuracy(
    actual: tuple[tuple[str, str, int | None], ...],
    expected: tuple[tuple[str, str, int | None], ...],
) -> float:
    """Measure PDF page boundaries without treating page-internal whitespace as a split."""
    if not actual and not expected:
        return 1.0
    denominator = max(len(actual), len(expected))
    if denominator == 0:
        return 0.0
    matches = sum(
        content_view(actual_segment[0]) == content_view(expected_segment[0])
        for actual_segment, expected_segment in zip(actual, expected, strict=False)
    )
    return matches / denominator


def segment_provenance_accuracy(
    actual: tuple[tuple[str, str, int | None], ...],
    expected: tuple[tuple[str, str, int | None], ...],
) -> float:
    """Measure exact heading and page provenance for aligned segments."""
    if not actual and not expected:
        return 1.0
    denominator = max(len(actual), len(expected))
    if denominator == 0:
        return 0.0
    matches = sum(
        actual_segment[1:] == expected_segment[1:]
        for actual_segment, expected_segment in zip(actual, expected, strict=False)
    )
    return matches / denominator


def _ngrams(text: str, width: int) -> Counter[str]:
    if not text:
        return Counter()
    effective_width = min(width, len(text))
    return Counter(text[index : index + effective_width] for index in range(len(text) - effective_width + 1))


def _bit_parallel_levenshtein(pattern: str, text: str) -> int:
    """Compute exact Levenshtein distance with Myers' arbitrary-width bit vectors."""
    if not pattern:
        return len(text)
    masks: dict[str, int] = {}
    for index, character in enumerate(pattern):
        masks[character] = masks.get(character, 0) | (1 << index)
    positive = ~0
    negative = 0
    score = len(pattern)
    highest = 1 << (len(pattern) - 1)
    for character in text:
        matches = masks.get(character, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        positive_delta = negative | ~(horizontal | positive)
        negative_delta = positive & horizontal
        if positive_delta & highest:
            score += 1
        elif negative_delta & highest:
            score -= 1
        positive_delta = (positive_delta << 1) | 1
        negative_delta <<= 1
        positive = negative_delta | ~(vertical | positive_delta)
        negative = positive_delta & vertical
    return score
