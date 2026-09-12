"""Tests for frozen manual document-extractor truth."""

from __future__ import annotations

from evals.document_extractors.manual_truth import load_pdf_quality_round_2_truth


def test_pdf_quality_round_2_truth_is_complete_and_pinned() -> None:
    truths = load_pdf_quality_round_2_truth()

    assert len(truths) == 6
    assert sum(truth.selection_scope for truth in truths) == 4
    assert len({truth.sample_id for truth in truths}) == len(truths)
    assert all(len(truth.sha256) == 64 for truth in truths)
    assert all(truth.anchors and truth.page_anchors for truth in truths)
