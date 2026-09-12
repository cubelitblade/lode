"""Manually adjudicated truth for disputed public extractor samples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
PDF_QUALITY_ROUND_2_TRUTH = ROOT / "evals" / "document_extractors" / "truth" / "pdf-quality-round-2.json"


@dataclass(frozen=True, slots=True)
class ManualPdfTruth:
    """Human-readable content and order observed in one rendered PDF."""

    sample_id: str
    category: str
    sha256: str
    selection_scope: bool
    expected_text: str | None
    anchors: tuple[str, ...]
    page_anchors: tuple[tuple[str, ...], ...]
    note: str


class ManualPdfTruthEntry(TypedDict):
    sample_id: str
    category: str
    sha256: str
    selection_scope: bool
    anchors: list[str]
    page_anchors: list[list[str]]
    note: str
    expected_text: NotRequired[str]


def load_pdf_quality_round_2_truth() -> tuple[ManualPdfTruth, ...]:
    """Load the frozen manual adjudications used by PDF Quality round 2."""
    payload: object = json.loads(PDF_QUALITY_ROUND_2_TRUTH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"invalid manual truth file: {PDF_QUALITY_ROUND_2_TRUTH}")
    entries = cast(list[ManualPdfTruthEntry], payload)
    return tuple(
        ManualPdfTruth(
            sample_id=entry["sample_id"],
            category=entry["category"],
            sha256=entry["sha256"],
            selection_scope=entry["selection_scope"],
            expected_text=entry.get("expected_text"),
            anchors=tuple(entry["anchors"]),
            page_anchors=tuple(tuple(page) for page in entry["page_anchors"]),
            note=entry["note"],
        )
        for entry in entries
    )


def verify_manual_truth_document(truth: ManualPdfTruth, path: Path) -> None:
    """Refuse to adjudicate a file that differs from the visually reviewed bytes."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != truth.sha256:
        raise ValueError(f"SHA-256 mismatch for manual truth {truth.sample_id}: {actual}")
