"""Tests for frozen manual document-extractor truth."""

from __future__ import annotations

import json
from pathlib import Path

from evals.document_extractors.corpus import DOC_MANIFEST
from evals.document_extractors.fixtures import create_doc_invalid_fixtures, make_doc_public_fixture
from evals.document_extractors.manual_truth import load_pdf_quality_round_2_truth


def test_doc_smoke_corpus_manifest_and_truth_are_pinned(tmp_path: Path) -> None:
    manifest = json.loads(DOC_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["license"] == "MIT"
    assert len(manifest["files"]) == 3
    assert all(item["name"].endswith(".doc") for item in manifest["files"])
    public_fixtures = [
        make_doc_public_fixture(tmp_path / item["name"], fixture_id=f"public-{index:03d}")
        for index, item in enumerate(manifest["files"], start=1)
    ]
    assert all(fixture.valid for fixture in public_fixtures)

    valid_path = tmp_path / "valid.doc"
    valid_path.write_bytes(b"synthetic source bytes")
    invalid = create_doc_invalid_fixtures(tmp_path / "invalid", valid_path)
    assert [fixture.path.suffix for fixture in invalid] == [".doc", ".doc"]
    assert all(not fixture.valid for fixture in invalid)


def test_pdf_quality_round_2_truth_is_complete_and_pinned() -> None:
    truths = load_pdf_quality_round_2_truth()

    assert len(truths) == 6
    assert sum(truth.selection_scope for truth in truths) == 4
    assert len({truth.sample_id for truth in truths}) == len(truths)
    assert all(len(truth.sha256) == 64 for truth in truths)
    assert all(truth.anchors and truth.page_anchors for truth in truths)
