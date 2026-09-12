"""DOCX canonical Markdown quality evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.document_extractors.corpus import CorpusDocument, load_docx_public_corpus
from evals.document_extractors.fixtures import Fixture, create_docx_quality_fixtures
from evals.document_extractors.metrics import (
    anchor_order_accuracy,
    content_view,
    markdown_content_projection,
    markdown_delimiters_balanced,
    markdown_heading_sequence,
    ngram_prf,
    segment_provenance_accuracy,
)
from evals.document_extractors.operational import (
    _build_production_runner,  # pyright: ignore[reportPrivateUsage]
    _timed_extract,  # pyright: ignore[reportPrivateUsage]
)
from evals.document_extractors.run import (
    ROOT,
    _python_extract,  # pyright: ignore[reportPrivateUsage]
    _rustc_version,  # pyright: ignore[reportPrivateUsage]
)


def _expected_segments(fixture: Fixture) -> tuple[tuple[str, str, int | None], ...]:
    return tuple((segment.text, segment.heading, segment.page) for segment in fixture.expected_segments)


def _actual_segments(payload: dict[str, Any]) -> tuple[tuple[str, str, int | None], ...]:
    return tuple(
        (str(segment["text"]), str(segment["heading"]), segment.get("page")) for segment in payload.get("segments", [])
    )


def _projected_segment_accuracy(payload: dict[str, Any], fixture: Fixture) -> tuple[float, float]:
    actual = _actual_segments(payload)
    expected = _expected_segments(fixture)
    if not actual and not expected:
        return 1.0, 1.0
    denominator = max(len(actual), len(expected))
    matches = sum(
        content_view(markdown_content_projection(actual_item[0])) == content_view(expected_item[0])
        for actual_item, expected_item in zip(actual, expected, strict=False)
    )
    boundary = matches / denominator if denominator else 0.0
    provenance = segment_provenance_accuracy(
        tuple((markdown_content_projection(text), heading, page) for text, heading, page in actual),
        expected,
    )
    return boundary, provenance


def _generated_case(fixture: Fixture, payload: dict[str, Any]) -> dict[str, Any]:
    markdown = str(payload.get("text", "")) if payload.get("status") == "ok" else ""
    projected = markdown_content_projection(markdown)
    _, _, f1 = ngram_prf(projected, fixture.expected_text)
    boundary, provenance = _projected_segment_accuracy(payload, fixture)
    headings = markdown_heading_sequence(markdown)
    expected_headings = fixture.expected_markdown_headings
    heading_accuracy = (
        sum(left == right for left, right in zip(headings, expected_headings, strict=False))
        / max(len(headings), len(expected_headings))
        if headings or expected_headings
        else 1.0
    )
    return {
        "sample_id": fixture.fixture_id,
        "status": str(payload.get("status", "error")),
        "process_status": str(payload.get("process_status", "error")),
        "projected_f1": f1 if markdown else None,
        "anchor_order": anchor_order_accuracy(projected, fixture.anchors) if markdown else None,
        "segment_boundaries": boundary if markdown else None,
        "provenance": provenance if markdown else None,
        "heading_accuracy": heading_accuracy if markdown else None,
        "delimiter_balanced": markdown_delimiters_balanced(markdown) if markdown else None,
        "segment_join_ok": "\n\n".join(str(item["text"]) for item in payload.get("segments", [])) == markdown,
        "elapsed_ns": int(payload.get("elapsed_ns", 0)),
        "error": payload.get("error"),
    }


def _public_case(sample_id: str, payload: dict[str, Any], reference: str) -> dict[str, Any]:
    markdown = str(payload.get("text", "")) if payload.get("status") == "ok" else ""
    projected = markdown_content_projection(markdown)
    _, _, f1 = ngram_prf(projected, reference) if markdown and reference else (None, None, None)
    return {
        "sample_id": sample_id,
        "status": str(payload.get("status", "error")),
        "projected_f1": f1,
        "delimiter_balanced": markdown_delimiters_balanced(markdown) if markdown else None,
        "heading_count": len(markdown_heading_sequence(markdown)) if markdown else 0,
        "elapsed_ns": int(payload.get("elapsed_ns", 0)),
        "error": payload.get("error"),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _report_markdown(report: dict[str, Any]) -> str:
    generated = report["generated"]
    public = report["public"]
    public_cases: list[dict[str, Any]] = public["cases"]
    lines = [
        "# DOCX canonical Markdown quality report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Annotated fixtures: `{len(generated)}`",
        f"- Public corpus: `{len(public['cases'])}`",
        "",
        "## Annotated fixtures",
        "",
        "| Fixture | Status | Projected F1 | Anchor order | Boundaries | Provenance | Headings | Balanced | Joined |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in generated:
        lines.append(
            f"| {case['sample_id']} | {case['status']} | {_optional_score(case['projected_f1'])} | "
            f"{_optional_score(case['anchor_order'])} | {_optional_score(case['segment_boundaries'])} | "
            f"{_optional_score(case['provenance'])} | {_optional_score(case['heading_accuracy'])} | "
            f"{'—' if case['delimiter_balanced'] is None else ('PASS' if case['delimiter_balanced'] else 'FAIL')} | "
            f"{'PASS' if case['segment_join_ok'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Public corpus diagnostics",
            "",
            f"- Successful extraction: `{public['success_rate']:.3f}`",
            f"- Mean projected F1 against Python baseline: `{public['mean_projected_f1'] or 0.0:.3f}`",
            f"- Balanced Markdown: `{public['balanced_rate']:.3f}`",
            "",
            "### Lowest projected-F1 samples",
            "",
            "| Sample | Status | Projected F1 | Balanced | Headings | Error |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    public_with_scores = [case for case in public_cases if case["projected_f1"] is not None]
    public_with_scores.sort(key=lambda case: float(case["projected_f1"]))
    for case in public_with_scores[:10]:
        error = str(case["error"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {case['sample_id']} | {case['status']} | {_optional_score(case['projected_f1'])} | "
            f"{'—' if case['delimiter_balanced'] is None else ('PASS' if case['delimiter_balanced'] else 'FAIL')} | "
            f"{case['heading_count']} | {error} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )
    failed_structure = [
        case["sample_id"]
        for case in generated
        if case["heading_accuracy"] != 1.0 or case["segment_boundaries"] != 1.0 or case["provenance"] != 1.0
    ]
    if failed_structure:
        lines.append(
            "- Heading 0/Title fixtures currently fail structural parity: "
            + ", ".join(f"`{sample_id}`" for sample_id in failed_structure)
            + "."
        )
    else:
        lines.append("- All annotated heading and segment structure checks passed.")
    unbalanced_public = [case["sample_id"] for case in public_cases if case["delimiter_balanced"] is False]
    if unbalanced_public:
        lines.append(
            "- Delimiter parity reported a diagnostic failure in public sample(s): "
            + ", ".join(f"`{sample_id}`" for sample_id in unbalanced_public)
            + "; this check is intentionally conservative."
        )
    lines.extend(
        [
            "",
            "The public corpus has no manually adjudicated Markdown truth in this round; its projection and "
            "delimiter results are diagnostic only. Private paths and text are omitted.",
            "",
            "## Gate",
            "",
            "The annotated gate requires successful extraction, projected F1 >= 0.98, anchor order >= 0.97, "
            "segment boundary/provenance >= 0.95, exact heading sequence, balanced delimiters, and lossless "
            "segment joining. Public success must be no more than one percentage point below the Python baseline.",
            "",
            f"- Result: **{'PASS' if report['quality_pass'] else 'FAIL'}**",
            "",
        ]
    )
    return "\n".join(lines)


def run_docx_markdown_quality(*, public_corpus_dir: Path | None = None, private_corpus_dir: Path | None = None) -> Path:
    del private_corpus_dir  # Private paths are intentionally not used for this first public baseline.
    executable, build_seconds = _build_production_runner()
    fixtures: list[Fixture]
    public_metadata: Any
    public_documents: list[CorpusDocument]
    public_metadata, public_documents = load_docx_public_corpus(public_corpus_dir)
    with tempfile.TemporaryDirectory(prefix="lode-docx-markdown-quality-") as temporary:
        fixtures = [fixture for fixture in create_docx_quality_fixtures(Path(temporary)) if fixture.valid]
        generated_cases = [_generated_case(fixture, _timed_extract(executable, fixture)) for fixture in fixtures]
        public_cases = []
        for document in public_documents:
            payload = _timed_extract(executable, Fixture(document.sample_id, document.path, True))
            reference = _python_extract(document.path)
            public_cases.append(_public_case(document.sample_id, payload, reference.text))

    projected_f1 = [case["projected_f1"] for case in generated_cases if case["projected_f1"] is not None]
    anchor_order = [case["anchor_order"] for case in generated_cases if case["anchor_order"] is not None]
    boundaries = [case["segment_boundaries"] for case in generated_cases if case["segment_boundaries"] is not None]
    provenance = [case["provenance"] for case in generated_cases if case["provenance"] is not None]
    heading_accuracy = [case["heading_accuracy"] for case in generated_cases if case["heading_accuracy"] is not None]
    public_success = sum(case["status"] == "ok" for case in public_cases) / len(public_cases) if public_cases else 0.0
    public_f1 = [case["projected_f1"] for case in public_cases if case["projected_f1"] is not None]
    balanced = [case["delimiter_balanced"] for case in public_cases if case["delimiter_balanced"] is not None]
    quality_pass = (
        all(case["status"] == "ok" and case["process_status"] == "ok" for case in generated_cases)
        and min(projected_f1, default=0.0) >= 0.98
        and min(anchor_order, default=0.0) >= 0.97
        and min(boundaries, default=0.0) >= 0.95
        and min(provenance, default=0.0) >= 0.95
        and min(heading_accuracy, default=0.0) == 1.0
        and all(case["delimiter_balanced"] and case["segment_join_ok"] for case in generated_cases)
        and public_success >= 0.99
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-docx-markdown-quality")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "docx",
        "round": "quality-markdown",
        "environment": {"platform": platform.platform(), "rustc": _rustc_version()},
        "build_seconds": build_seconds,
        "public_corpus": asdict(public_metadata),
        "generated": generated_cases,
        "public": {
            "cases": public_cases,
            "success_rate": public_success,
            "mean_projected_f1": _mean(public_f1),
            "balanced_rate": sum(balanced) / len(balanced) if balanced else 0.0,
        },
        "quality_pass": quality_pass,
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _report_markdown(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DOCX canonical Markdown quality evaluation")
    parser.add_argument("--public-corpus-dir", type=Path)
    args = parser.parse_args()
    run_docx_markdown_quality(public_corpus_dir=args.public_corpus_dir)


if __name__ == "__main__":
    main()
