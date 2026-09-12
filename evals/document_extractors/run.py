"""Run staged document extractor evaluations.

Selection rounds compare isolated candidate adapters and write both JSON and
Markdown reports under the locally excluded ``.ai/process`` tree.

Usage:
    uv run python -m evals.document_extractors.run --format docx --round smoke
    uv run python -m evals.document_extractors.run --format doc --round smoke
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.document_extractors.corpus import (
    CorpusDocument,
    load_doc_public_corpus,
    load_docx_public_corpus,
    load_pdf_public_corpus,
    load_private_docx_corpus,
    load_private_pdf_corpus,
)
from evals.document_extractors.fixtures import (
    Fixture,
    create_doc_invalid_fixtures,
    create_docx_quality_fixtures,
    create_docx_smoke_fixtures,
    create_pdf_quality_fixtures,
    create_pdf_smoke_fixtures,
    make_doc_public_fixture,
)
from evals.document_extractors.manual_truth import (
    ManualPdfTruth,
    load_pdf_quality_round_2_truth,
    verify_manual_truth_document,
)
from evals.document_extractors.metrics import (
    anchor_order_accuracy,
    invalid_unicode_character_count,
    markdown_delimiters_balanced,
    markdown_heading_accuracy,
    markdown_plain_text,
    markdown_table_count,
    ngram_prf,
    normalized_edit_similarity,
    segment_boundary_accuracy,
    segment_content_boundary_accuracy,
    segment_provenance_accuracy,
    table_row_accuracy,
)
from lode.ingestion.extract import extract_document

ROOT = Path(__file__).resolve().parents[2]
RUST_MANIFEST = ROOT / "evals" / "document_extractors" / "rust-runner" / "Cargo.toml"
RUST_TARGET_DIR = Path(tempfile.gettempdir()) / "lode-document-extractor-target"
DOCX_RUST_CANDIDATES = ("office_oxide", "rwml", "docx_rs", "rs_docx")
DOC_RUST_CANDIDATES = ("office_oxide", "rwml")
PDF_RUST_CANDIDATES = ("pdf_oxide", "pdf_extract")
ALLOWED_LICENSES = {"MIT", "MIT OR Apache-2.0", "Apache-2.0 OR MIT"}
RUST_TOOLCHAIN = os.environ.get("LODE_EVAL_RUST_TOOLCHAIN")


@dataclass(frozen=True, slots=True)
class SegmentResult:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    candidate: str
    format: str
    status: str
    text: str
    markdown: str | None
    segments: tuple[SegmentResult, ...]
    warnings: tuple[str, ...]
    elapsed_ns: int
    error: str | None = None
    process_status: str = "ok"


@dataclass(frozen=True, slots=True)
class CaseScore:
    fixture_id: str
    expected_status: str
    actual_status: str
    ngram_precision: float | None
    ngram_recall: float | None
    ngram_f1: float | None
    edit_similarity: float | None
    anchor_order: float | None
    table_row_accuracy: float | None
    elapsed_ns: int
    error: str | None


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate: str
    license: str
    direct_license_allowed: bool
    license_allowed: bool
    valid_parsed: int
    valid_total: int
    invalid_rejected: int
    invalid_total: int
    minimum_anchor_order: float
    mean_ngram_f1: float
    mean_table_row_accuracy: float | None
    smoke_pass: bool
    cases: tuple[CaseScore, ...]


@dataclass(frozen=True, slots=True)
class QualityCase:
    sample_id: str
    source: str
    reference_status: str
    actual_status: str
    ngram_f1: float | None
    edit_similarity: float | None
    anchor_order: float | None
    segment_boundaries: float | None
    provenance: float | None
    table_rows: float | None
    process_status: str
    error: str | None
    diff_excerpt: str | None


@dataclass(frozen=True, slots=True)
class QualitySummary:
    candidate: str
    annotated_success_rate: float
    annotated_min_ngram_f1: float
    annotated_min_anchor_order: float
    segment_boundary_accuracy: float
    provenance_accuracy: float
    table_row_accuracy: float
    public_success_rate: float
    public_reference_success_rate: float
    public_mean_ngram_f1: float
    private_samples: int
    private_success_rate: float | None
    crashes_or_timeouts: int
    quality_score: float
    quality_pass: bool
    cases: tuple[QualityCase, ...]


@dataclass(frozen=True, slots=True)
class DocQualityCase:
    fixture_id: str
    expected_status: str
    actual_status: str
    ngram_f1: float | None
    edit_similarity: float | None
    anchor_order: float | None
    segment_boundaries: float | None
    expected_markdown_tables: int
    actual_markdown_tables: int | None
    markdown_delimiters_balanced: bool | None
    process_status: str
    error: str | None


@dataclass(frozen=True, slots=True)
class DocQualitySummary:
    candidate: str
    license: str
    license_allowed: bool
    valid_success_rate: float
    invalid_rejection_rate: float
    minimum_ngram_f1: float
    minimum_anchor_order: float
    segment_boundary_accuracy: float
    markdown_table_accuracy: float
    quality_score: float
    quality_pass: bool
    crashes_or_timeouts: int
    cases: tuple[DocQualityCase, ...]


@dataclass(frozen=True, slots=True)
class PdfSmokeCase:
    fixture_id: str
    selection_scope: bool
    expected_status: str
    actual_status: str
    anchor_order: float | None
    segment_boundaries: float | None
    provenance: float | None
    markdown_emitted: bool
    markdown_delimiters_balanced: bool | None
    process_status: str
    error: str | None


@dataclass(frozen=True, slots=True)
class PdfSmokeSummary:
    candidate: str
    license: str
    license_allowed: bool
    valid_parsed: int
    valid_total: int
    invalid_rejected: int
    invalid_total: int
    minimum_anchor_order: float
    boundary_accuracy: float
    provenance_accuracy: float
    markdown_coverage: float
    markdown_delimiters_balanced: bool | None
    smoke_pass: bool
    cases: tuple[PdfSmokeCase, ...]


@dataclass(frozen=True, slots=True)
class PdfQualityCase:
    sample_id: str
    source: str
    selection_scope: bool
    reference_status: str
    actual_status: str
    ngram_f1: float | None
    edit_similarity: float | None
    anchor_order: float | None
    segment_boundaries: float | None
    provenance: float | None
    markdown_precision: float | None
    markdown_f1: float | None
    markdown_order: float | None
    markdown_headings: float | None
    markdown_balanced: bool | None
    process_status: str
    error: str | None
    diff_excerpt: str | None


@dataclass(frozen=True, slots=True)
class PdfQualitySummary:
    candidate: str
    annotated_success_rate: float
    annotated_min_ngram_f1: float
    annotated_min_anchor_order: float
    segment_boundary_accuracy: float
    provenance_accuracy: float
    public_success_rate: float
    public_reference_success_rate: float
    public_mean_ngram_f1: float
    markdown_coverage: float
    markdown_mean_precision: float | None
    markdown_mean_f1: float | None
    markdown_min_order: float | None
    markdown_heading_accuracy: float | None
    markdown_balanced: bool | None
    private_samples: int
    private_success_rate: float | None
    crashes_or_timeouts: int
    quality_score: float
    quality_pass: bool
    cases: tuple[PdfQualityCase, ...]


@dataclass(frozen=True, slots=True)
class PdfAdjudicationCase:
    sample_id: str
    category: str
    selection_scope: bool
    actual_status: str
    strict_ngram_f1: float | None
    compatibility_ngram_f1: float | None
    strict_anchor_order: float
    compatibility_anchor_order: float
    strict_page_anchor_accuracy: float
    compatibility_page_anchor_accuracy: float
    invalid_unicode_characters: int
    control_characters: int
    markdown_anchor_order: float | None
    markdown_balanced: bool | None
    case_pass: bool
    note: str
    error: str | None


@dataclass(frozen=True, slots=True)
class PdfAdjudicationSummary:
    candidate: str
    selection_passed: int
    selection_total: int
    diagnostic_passed: int
    diagnostic_total: int
    strict_mean_ngram_f1: float
    compatibility_mean_ngram_f1: float
    focused_quality_pass: bool
    cases: tuple[PdfAdjudicationCase, ...]


def _python_extract(path: Path) -> ExtractionResult:
    started = time.perf_counter_ns()
    try:
        segments = extract_document(path.read_bytes(), path.suffix)
        if segments is None:
            raise RuntimeError("supported DOCX fixture was reported as unsupported")
        converted = tuple(SegmentResult(text=s.text, heading=s.heading, page=s.page) for s in segments)
        return ExtractionResult(
            candidate="python",
            format=path.suffix.lower().lstrip("."),
            status="ok",
            text="\n\n".join(segment.text for segment in converted),
            markdown=None,
            segments=converted,
            warnings=(),
            elapsed_ns=time.perf_counter_ns() - started,
        )
    except Exception as exc:
        return ExtractionResult(
            candidate="python",
            format=path.suffix.lower().lstrip("."),
            status="error",
            text="",
            markdown=None,
            segments=(),
            warnings=(),
            elapsed_ns=time.perf_counter_ns() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _build_rust_runner() -> tuple[Path, float]:
    started = time.perf_counter()
    environment = {**os.environ, "CARGO_TARGET_DIR": str(RUST_TARGET_DIR)}
    subprocess.run(
        _cargo_command("build", "--release", "--manifest-path", str(RUST_MANIFEST)),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    executable = (
        RUST_TARGET_DIR
        / "release"
        / ("document-extractor-rust-runner.exe" if os.name == "nt" else "document-extractor-rust-runner")
    )
    return executable, time.perf_counter() - started


def _rust_extract(executable: Path, candidate: str, path: Path) -> ExtractionResult:
    format_name = path.suffix.lower().lstrip(".")
    try:
        completed = subprocess.run(
            [str(executable), candidate, format_name, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return ExtractionResult(
            candidate=candidate,
            format=format_name,
            status="error",
            text="",
            markdown=None,
            segments=(),
            warnings=(),
            elapsed_ns=0,
            error="candidate timed out after 15 seconds",
            process_status="timeout",
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return ExtractionResult(
            candidate=candidate,
            format=format_name,
            status="error",
            text="",
            markdown=None,
            segments=(),
            warnings=(),
            elapsed_ns=0,
            error=detail,
            process_status="crash",
        )
    try:
        payload = json.loads(completed.stdout)
        return ExtractionResult(
            candidate=str(payload["candidate"]),
            format=str(payload["format"]),
            status=str(payload["status"]),
            text=str(payload["text"]),
            markdown=None if payload.get("markdown") is None else str(payload["markdown"]),
            segments=tuple(SegmentResult(**segment) for segment in payload["segments"]),
            warnings=tuple(str(item) for item in payload["warnings"]),
            elapsed_ns=int(payload["elapsed_ns"]),
            error=None if payload.get("error") is None else str(payload["error"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return ExtractionResult(
            candidate=candidate,
            format=format_name,
            status="error",
            text="",
            markdown=None,
            segments=(),
            warnings=(),
            elapsed_ns=0,
            error=f"invalid runner output: {exc}; stdout={completed.stdout!r}",
            process_status="protocol_error",
        )


def _candidate_metadata() -> dict[str, tuple[str, str]]:
    environment = {**os.environ, "CARGO_TARGET_DIR": str(RUST_TARGET_DIR)}
    completed = subprocess.run(
        _cargo_command("metadata", "--format-version", "1", "--manifest-path", str(RUST_MANIFEST)),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(completed.stdout)["packages"]
    names = {"office_oxide", "rwml", "docx-rs", "rs-docx", "pdf_oxide", "pdf-extract"}
    return {
        str(package["name"]): (str(package["version"]), str(package.get("license") or "UNKNOWN"))
        for package in packages
        if package["name"] in names
    }


def _cargo_command(*args: str) -> list[str]:
    command = ["cargo"]
    if RUST_TOOLCHAIN:
        command.append(f"+{RUST_TOOLCHAIN}")
    command.extend(args)
    return command


def _rustc_version() -> str:
    command = ["rustc", "--version"]
    if RUST_TOOLCHAIN:
        command = ["rustup", "run", RUST_TOOLCHAIN, *command]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def _score(
    candidate: str,
    results: list[tuple[Fixture, ExtractionResult]],
    license_name: str,
    *,
    dependency_licenses_allowed: bool,
) -> CandidateSummary:
    cases: list[CaseScore] = []
    valid_parsed = 0
    invalid_rejected = 0
    anchor_scores: list[float] = []
    f1_scores: list[float] = []
    table_scores: list[float] = []
    process_ok = True
    for fixture, result in results:
        process_ok = process_ok and result.process_status == "ok"
        if fixture.valid and result.status == "ok":
            valid_parsed += 1
            precision, recall, f1 = ngram_prf(result.text, fixture.expected_text)
            edit = normalized_edit_similarity(result.text, fixture.expected_text)
            anchors = anchor_order_accuracy(result.text, fixture.anchors)
            table_rows = table_row_accuracy(result.text, fixture.table_rows) if fixture.table_rows else None
            anchor_scores.append(anchors)
            f1_scores.append(f1)
            if table_rows is not None:
                table_scores.append(table_rows)
        else:
            precision = recall = f1 = edit = anchors = table_rows = None
        if not fixture.valid and result.status == "error":
            invalid_rejected += 1
        cases.append(
            CaseScore(
                fixture_id=fixture.fixture_id,
                expected_status="ok" if fixture.valid else "error",
                actual_status=result.status,
                ngram_precision=precision,
                ngram_recall=recall,
                ngram_f1=f1,
                edit_similarity=edit,
                anchor_order=anchors,
                table_row_accuracy=table_rows,
                elapsed_ns=result.elapsed_ns,
                error=result.error,
            )
        )
    valid_total = sum(fixture.valid for fixture, _ in results)
    invalid_total = len(results) - valid_total
    minimum_anchor_order = min(anchor_scores, default=0.0)
    mean_ngram_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_table_row_accuracy = sum(table_scores) / len(table_scores) if table_scores else None
    direct_license_allowed = candidate == "python" or license_name in ALLOWED_LICENSES
    license_allowed = direct_license_allowed and (candidate == "python" or dependency_licenses_allowed)
    smoke_pass = (
        process_ok
        and license_allowed
        and valid_parsed == valid_total
        and invalid_rejected == invalid_total
        and minimum_anchor_order == 1.0
    )
    return CandidateSummary(
        candidate=candidate,
        license=license_name,
        direct_license_allowed=direct_license_allowed,
        license_allowed=license_allowed,
        valid_parsed=valid_parsed,
        valid_total=valid_total,
        invalid_rejected=invalid_rejected,
        invalid_total=invalid_total,
        minimum_anchor_order=minimum_anchor_order,
        mean_ngram_f1=mean_ngram_f1,
        mean_table_row_accuracy=mean_table_row_accuracy,
        smoke_pass=smoke_pass,
        cases=tuple(cases),
    )


def _license_audit() -> tuple[bool, str]:
    completed = subprocess.run(
        [
            "cargo",
            "deny",
            "--manifest-path",
            str(RUST_MANIFEST),
            "--config",
            str(ROOT / "lode-rs" / "deny.toml"),
            "check",
            "licenses",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return completed.returncode == 0, output


def _markdown_report(report: dict[str, Any]) -> str:
    format_label = "legacy DOC" if report["format"] == "doc" else "DOCX"
    lines = [
        f"# {format_label} extractor smoke report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Platform: `{report['environment']['platform']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Rust runner build: `{report['build_seconds']:.2f}s`",
        "",
        "## Summary",
        "",
        "| Candidate | License | License gate | Valid | Invalid rejected | Min anchor order | "
        "Mean 3-gram F1 | Table rows | Smoke |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['candidate']} | {summary['license']} | "
            f"{'PASS' if summary['license_allowed'] else 'FAIL'} | "
            f"{summary['valid_parsed']}/{summary['valid_total']} | "
            f"{summary['invalid_rejected']}/{summary['invalid_total']} | "
            f"{summary['minimum_anchor_order']:.3f} | {summary['mean_ngram_f1']:.3f} | "
            f"{_optional_score(summary['mean_table_row_accuracy'])} | "
            f"{'PASS' if summary['smoke_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Cases", ""])
    for summary in report["summaries"]:
        lines.extend(
            [
                f"### {summary['candidate']}",
                "",
                "| Fixture | Expected | Actual | Anchor order | 3-gram F1 | Edit similarity | Table rows | Error |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for case in summary["cases"]:
            anchor = "—" if case["anchor_order"] is None else f"{case['anchor_order']:.3f}"
            f1 = "—" if case["ngram_f1"] is None else f"{case['ngram_f1']:.3f}"
            edit = "—" if case["edit_similarity"] is None else f"{case['edit_similarity']:.3f}"
            table_rows = _optional_score(case["table_row_accuracy"])
            error = (case["error"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {case['fixture_id']} | {case['expected_status']} | {case['actual_status']} | "
                f"{anchor} | {f1} | {edit} | {table_rows} | {error} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Dependency license audit",
            "",
            f"- Result: `{'PASS' if report['license_audit']['passed'] else 'FAIL'}`",
            "",
            "```text",
            report["license_audit"]["output"],
            "```",
            "",
        ]
    )
    lines.extend(["## Observed valid text", ""])
    for candidate, cases in report["results"].items():
        for case in cases:
            result = case["result"]
            if case["valid"] and result["status"] == "ok":
                lines.extend(
                    [
                        f"### {candidate} / {case['fixture_id']}",
                        "",
                        "```text",
                        result["text"],
                        "```",
                        "",
                    ]
                )
    lines.extend(
        [
            "## Gate",
            "",
            "Smoke passes only when the candidate has an allowed license, parses every valid fixture, "
            "rejects every malformed fixture without crashing, and preserves all anchors in order.",
            "",
            "This report is a feasibility gate, not the final quality selection.",
            "",
        ]
    )
    return "\n".join(lines)


def _optional_score(score: float | None) -> str:
    return "—" if score is None else f"{score:.3f}"


def run_docx_smoke() -> Path:
    executable, build_seconds = _build_rust_runner()
    metadata = _candidate_metadata()
    with tempfile.TemporaryDirectory(prefix="lode-docx-smoke-") as temporary:
        fixtures = create_docx_smoke_fixtures(Path(temporary))
        all_results: dict[str, list[tuple[Fixture, ExtractionResult]]] = {
            "python": [(fixture, _python_extract(fixture.path)) for fixture in fixtures]
        }
        for candidate in DOCX_RUST_CANDIDATES:
            all_results[candidate] = [
                (fixture, _rust_extract(executable, candidate, fixture.path)) for fixture in fixtures
            ]

    dependency_licenses_allowed, license_output = _license_audit()
    summaries: list[CandidateSummary] = []
    summaries.append(
        _score(
            "python",
            all_results["python"],
            "baseline",
            dependency_licenses_allowed=dependency_licenses_allowed,
        )
    )
    package_names = {
        "office_oxide": "office_oxide",
        "rwml": "rwml",
        "docx_rs": "docx-rs",
        "rs_docx": "rs-docx",
    }
    versions: dict[str, str] = {"python": "python-docx 1.2.0 / PyMuPDF 1.28.2"}
    for candidate in DOCX_RUST_CANDIDATES:
        package_name = package_names[candidate]
        version, license_name = metadata[package_name]
        versions[candidate] = version
        summaries.append(
            _score(
                candidate,
                all_results[candidate],
                license_name,
                dependency_licenses_allowed=dependency_licenses_allowed,
            )
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-docx-smoke")
    rustc = _rustc_version()
    serialized_results = {
        candidate: [
            {"fixture_id": fixture.fixture_id, "valid": fixture.valid, "result": asdict(result)}
            for fixture, result in results
        ]
        for candidate, results in all_results.items()
    }
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "docx",
        "round": "smoke",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": rustc,
        },
        "versions": versions,
        "build_seconds": build_seconds,
        "license_audit": {"passed": dependency_licenses_allowed, "output": license_output},
        "summaries": [asdict(summary) for summary in summaries],
        "results": serialized_results,
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def run_doc_smoke(*, public_corpus_dir: Path | None = None) -> Path:
    """Evaluate isolated native legacy DOC candidates before production integration."""
    executable, build_seconds = _build_rust_runner()
    public_metadata, public_documents = load_doc_public_corpus(public_corpus_dir)
    with tempfile.TemporaryDirectory(prefix="lode-doc-smoke-") as temporary:
        public_fixtures = [
            make_doc_public_fixture(document.path, fixture_id=document.sample_id) for document in public_documents
        ]
        invalid_fixtures = create_doc_invalid_fixtures(Path(temporary), public_documents[0].path)
        fixtures = [*public_fixtures, *invalid_fixtures]
        all_results = {
            candidate: [(fixture, _rust_extract(executable, candidate, fixture.path)) for fixture in fixtures]
            for candidate in DOC_RUST_CANDIDATES
        }

    dependency_licenses_allowed, license_output = _license_audit()
    package_names = {"office_oxide": "office_oxide", "rwml": "rwml"}
    metadata = _candidate_metadata()
    versions: dict[str, str] = {}
    summaries: list[CandidateSummary] = []
    for candidate in DOC_RUST_CANDIDATES:
        version, license_name = metadata[package_names[candidate]]
        versions[candidate] = version
        summaries.append(
            _score(
                candidate,
                all_results[candidate],
                license_name,
                dependency_licenses_allowed=dependency_licenses_allowed,
            )
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-doc-smoke")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "doc",
        "round": "smoke",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
        },
        "versions": versions,
        "build_seconds": build_seconds,
        "sample_counts": {
            "valid": len(public_fixtures),
            "invalid": len(invalid_fixtures),
        },
        "public_corpus": asdict(public_metadata),
        "license_audit": {"passed": dependency_licenses_allowed, "output": license_output},
        "summaries": [asdict(summary) for summary in summaries],
        "results": {
            candidate: [
                {"fixture_id": fixture.fixture_id, "valid": fixture.valid, "result": asdict(result)}
                for fixture, result in results
            ]
            for candidate, results in all_results.items()
        },
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def _score_doc_quality_candidate(
    candidate: str,
    results: list[tuple[Fixture, ExtractionResult]],
    license_name: str,
    *,
    dependency_licenses_allowed: bool,
) -> DocQualitySummary:
    cases: list[DocQualityCase] = []
    valid_success = 0
    invalid_rejected = 0
    f1_scores: list[float] = []
    anchor_scores: list[float] = []
    boundary_scores: list[float] = []
    table_scores: list[float] = []
    crashes_or_timeouts = 0
    for fixture, result in results:
        if result.process_status != "ok":
            crashes_or_timeouts += 1
        if fixture.valid and result.status == "ok":
            valid_success += 1
            _, _, f1 = ngram_prf(result.text, fixture.expected_text)
            edit = normalized_edit_similarity(result.text, fixture.expected_text)
            anchors = anchor_order_accuracy(result.text, fixture.anchors)
            boundaries = segment_boundary_accuracy(_segment_tuples(result), _expected_segment_tuples(fixture))
            actual_tables = markdown_table_count(result.markdown or "")
            table_accuracy = float(actual_tables == fixture.expected_markdown_table_count)
            f1_scores.append(f1)
            anchor_scores.append(anchors)
            boundary_scores.append(boundaries)
            table_scores.append(table_accuracy)
            balanced = markdown_delimiters_balanced(result.markdown) if result.markdown else None
        else:
            f1 = edit = anchors = boundaries = None
            actual_tables = None
            balanced = None
            if not fixture.valid and result.status == "error":
                invalid_rejected += 1
        cases.append(
            DocQualityCase(
                fixture_id=fixture.fixture_id,
                expected_status="ok" if fixture.valid else "error",
                actual_status=result.status,
                ngram_f1=f1,
                edit_similarity=edit,
                anchor_order=anchors,
                segment_boundaries=boundaries,
                expected_markdown_tables=fixture.expected_markdown_table_count,
                actual_markdown_tables=actual_tables,
                markdown_delimiters_balanced=balanced,
                process_status=result.process_status,
                error=result.error,
            )
        )
    valid_total = sum(fixture.valid for fixture, _ in results)
    invalid_total = len(results) - valid_total
    direct_license_allowed = license_name in ALLOWED_LICENSES
    license_allowed = direct_license_allowed and dependency_licenses_allowed
    valid_success_rate = valid_success / valid_total if valid_total else 0.0
    invalid_rejection_rate = invalid_rejected / invalid_total if invalid_total else 0.0
    minimum_f1 = min(f1_scores, default=0.0)
    minimum_anchor = min(anchor_scores, default=0.0)
    boundary_accuracy = _mean(boundary_scores)
    table_accuracy = _mean(table_scores)
    quality_score = 100 * _mean([_mean(f1_scores), _mean(anchor_scores), boundary_accuracy, table_accuracy])
    quality_pass = (
        license_allowed
        and crashes_or_timeouts == 0
        and valid_success == valid_total
        and invalid_rejected == invalid_total
        and minimum_f1 >= 0.98
        and minimum_anchor >= 0.97
        and boundary_accuracy >= 0.99
        and table_accuracy >= 0.99
    )
    return DocQualitySummary(
        candidate=candidate,
        license=license_name,
        license_allowed=license_allowed,
        valid_success_rate=valid_success_rate,
        invalid_rejection_rate=invalid_rejection_rate,
        minimum_ngram_f1=minimum_f1,
        minimum_anchor_order=minimum_anchor,
        segment_boundary_accuracy=boundary_accuracy,
        markdown_table_accuracy=table_accuracy,
        quality_score=quality_score,
        quality_pass=quality_pass,
        crashes_or_timeouts=crashes_or_timeouts,
        cases=tuple(cases),
    )


def _doc_quality_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Legacy DOC extractor quality report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Valid samples: `{report['sample_counts']['valid']}`",
        f"- Invalid samples: `{report['sample_counts']['invalid']}`",
        f"- Public source revision: `{report['public_corpus']['revision']}`",
        "",
        "## Summary",
        "",
        "| Candidate | License | Valid | Invalid rejected | Min F1 | Min order | Boundaries | "
        "Table structure | Score | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['candidate']} | {summary['license']} | {summary['valid_success_rate']:.3f} | "
            f"{summary['invalid_rejection_rate']:.3f} | {summary['minimum_ngram_f1']:.3f} | "
            f"{summary['minimum_anchor_order']:.3f} | {summary['segment_boundary_accuracy']:.3f} | "
            f"{summary['markdown_table_accuracy']:.3f} | {summary['quality_score']:.2f} | "
            f"{'PASS' if summary['quality_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Cases", ""])
    for summary in report["summaries"]:
        lines.extend(
            [
                f"### {summary['candidate']}",
                "",
                "| Fixture | Expected | Actual | F1 | Order | Boundaries | "
                "Tables expected/actual | Markdown balanced | Error |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for case in summary["cases"]:
            f1 = _optional_score(case["ngram_f1"])
            order = _optional_score(case["anchor_order"])
            boundaries = _optional_score(case["segment_boundaries"])
            tables = (
                "—"
                if case["actual_markdown_tables"] is None
                else (f"{case['expected_markdown_tables']}/{case['actual_markdown_tables']}")
            )
            balanced = (
                "—"
                if case["markdown_delimiters_balanced"] is None
                else ("yes" if case["markdown_delimiters_balanced"] else "no")
            )
            error = (case["error"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {case['fixture_id']} | {case['expected_status']} | {case['actual_status']} | {f1} | "
                f"{order} | {boundaries} | {tables} | {balanced} | {error} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate",
            "",
            "A candidate passes only with an allowed license, no crashes or timeouts, complete valid-file "
            "success, complete malformed-file rejection, minimum 3-gram F1 of 0.98, minimum anchor order "
            "of 0.97, segment boundaries of 0.99, and exact table-presence truth.",
            "",
            "Markdown is used here only as a structural diagnostic; it is not extraction truth.",
            "",
        ]
    )
    return "\n".join(lines)


def run_doc_quality(*, public_corpus_dir: Path | None = None) -> Path:
    """Compare isolated legacy DOC candidates against the frozen Quality truth."""
    executable, build_seconds = _build_rust_runner()
    public_metadata, public_documents = load_doc_public_corpus(public_corpus_dir)
    with tempfile.TemporaryDirectory(prefix="lode-doc-quality-") as temporary:
        public_fixtures = [
            make_doc_public_fixture(document.path, fixture_id=document.sample_id) for document in public_documents
        ]
        invalid_fixtures = create_doc_invalid_fixtures(Path(temporary), public_documents[0].path)
        fixtures = [*public_fixtures, *invalid_fixtures]
        all_results = {
            candidate: [(fixture, _rust_extract(executable, candidate, fixture.path)) for fixture in fixtures]
            for candidate in DOC_RUST_CANDIDATES
        }

    dependency_licenses_allowed, license_output = _license_audit()
    metadata = _candidate_metadata()
    package_names = {"office_oxide": "office_oxide", "rwml": "rwml"}
    summaries = [
        _score_doc_quality_candidate(
            candidate,
            all_results[candidate],
            metadata[package_names[candidate]][1],
            dependency_licenses_allowed=dependency_licenses_allowed,
        )
        for candidate in DOC_RUST_CANDIDATES
    ]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-doc-quality")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "doc",
        "round": "quality",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
        },
        "versions": {candidate: metadata[package_names[candidate]][0] for candidate in DOC_RUST_CANDIDATES},
        "build_seconds": build_seconds,
        "sample_counts": {"valid": len(public_fixtures), "invalid": len(invalid_fixtures)},
        "public_corpus": asdict(public_metadata),
        "license_audit": {"passed": dependency_licenses_allowed, "output": license_output},
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _doc_quality_markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def _run_all_candidates(
    executable: Path,
    documents: list[tuple[str, Path]],
) -> dict[str, dict[str, ExtractionResult]]:
    results = {
        "python": {sample_id: _python_extract(path) for sample_id, path in documents},
    }
    for candidate in DOCX_RUST_CANDIDATES:
        results[candidate] = {sample_id: _rust_extract(executable, candidate, path) for sample_id, path in documents}
    return results


def _segment_tuples(result: ExtractionResult) -> tuple[tuple[str, str, int | None], ...]:
    return tuple((segment.text, segment.heading, segment.page) for segment in result.segments)


def _expected_segment_tuples(fixture: Fixture) -> tuple[tuple[str, str, int | None], ...]:
    return tuple((segment.text, segment.heading, segment.page) for segment in fixture.expected_segments)


def _reference_anchors(text: str, *, limit: int = 20) -> tuple[str, ...]:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 3]
    if not lines:
        return ()
    if len(lines) <= limit:
        return tuple(lines)
    step = (len(lines) - 1) / (limit - 1)
    return tuple(lines[round(index * step)] for index in range(limit))


def _diff_excerpt(actual: str, expected: str, *, source: str) -> str | None:
    if source == "private" or actual == expected:
        return None
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="reference",
        tofile="candidate",
        lineterm="",
        n=2,
    )
    return "\n".join(list(diff)[:80])


def _mean(values: list[float], *, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _score_quality_candidate(
    candidate: str,
    generated: list[Fixture],
    corpus_documents: list[CorpusDocument],
    results: dict[str, dict[str, ExtractionResult]],
) -> QualitySummary:
    cases: list[QualityCase] = []
    generated_f1: list[float] = []
    generated_order: list[float] = []
    boundaries: list[float] = []
    provenance: list[float] = []
    table_rows: list[float] = []
    generated_success = 0
    public_success = 0
    public_reference_success = 0
    public_f1: list[float] = []
    private_success = 0
    private_total = 0
    crashes_or_timeouts = 0

    for fixture in generated:
        result = results[candidate][fixture.fixture_id]
        if result.process_status != "ok":
            crashes_or_timeouts += 1
        if fixture.valid and result.status == "ok":
            generated_success += 1
            _, _, f1 = ngram_prf(result.text, fixture.expected_text)
            edit = normalized_edit_similarity(result.text, fixture.expected_text)
            order = anchor_order_accuracy(result.text, fixture.anchors)
            expected_segments = _expected_segment_tuples(fixture)
            actual_segments = _segment_tuples(result)
            boundary = segment_boundary_accuracy(actual_segments, expected_segments)
            provenance_score = segment_provenance_accuracy(actual_segments, expected_segments)
            table_score = table_row_accuracy(result.text, fixture.table_rows) if fixture.table_rows else None
            generated_f1.append(f1)
            generated_order.append(order)
            boundaries.append(boundary)
            provenance.append(provenance_score)
            if table_score is not None:
                table_rows.append(table_score)
        else:
            f1 = edit = order = boundary = provenance_score = table_score = None
        cases.append(
            QualityCase(
                sample_id=fixture.fixture_id,
                source="annotated",
                reference_status="ok" if fixture.valid else "error",
                actual_status=result.status,
                ngram_f1=f1,
                edit_similarity=edit,
                anchor_order=order,
                segment_boundaries=boundary,
                provenance=provenance_score,
                table_rows=table_score,
                process_status=result.process_status,
                error=result.error,
                diff_excerpt=_diff_excerpt(result.text, fixture.expected_text, source="annotated"),
            )
        )

    for document in corpus_documents:
        result = results[candidate][document.sample_id]
        reference = results["python"][document.sample_id]
        if result.process_status != "ok":
            crashes_or_timeouts += 1
        if document.source == "public" and reference.status == "ok":
            public_reference_success += 1
        if document.source == "public" and result.status == "ok":
            public_success += 1
        if document.source == "private":
            private_total += 1
            if result.status == "ok":
                private_success += 1
        if result.status == "ok" and reference.status == "ok":
            _, _, f1 = ngram_prf(result.text, reference.text)
            edit = normalized_edit_similarity(result.text, reference.text)
            order = anchor_order_accuracy(result.text, _reference_anchors(reference.text))
            if document.source == "public":
                public_f1.append(f1)
        else:
            f1 = edit = order = None
        cases.append(
            QualityCase(
                sample_id=document.sample_id,
                source=document.source,
                reference_status=reference.status,
                actual_status=result.status,
                ngram_f1=f1,
                edit_similarity=edit,
                anchor_order=order,
                segment_boundaries=None,
                provenance=None,
                table_rows=None,
                process_status=result.process_status,
                error=result.error,
                diff_excerpt=(
                    _diff_excerpt(result.text, reference.text, source=document.source)
                    if result.status == "ok" and reference.status == "ok"
                    else None
                ),
            )
        )

    annotated_total = sum(fixture.valid for fixture in generated)
    public_total = sum(document.source == "public" for document in corpus_documents)
    annotated_success_rate = generated_success / annotated_total if annotated_total else 0.0
    public_success_rate = public_success / public_total if public_total else 0.0
    public_reference_success_rate = public_reference_success / public_total if public_total else 0.0
    private_success_rate = private_success / private_total if private_total else None
    boundary_score = _mean(boundaries)
    provenance_score = _mean(provenance)
    table_score = _mean(table_rows, default=1.0)
    structure_score = _mean([boundary_score, provenance_score, table_score])
    content_score = _mean([_mean(generated_f1), _mean(public_f1)])
    order_score = _mean(generated_order)
    robustness_score = _mean([annotated_success_rate, public_success_rate])
    quality_score = 100 * (0.40 * content_score + 0.25 * order_score + 0.25 * structure_score + 0.10 * robustness_score)
    exact_generated_structure = all(score == 1.0 for score in [*boundaries, *provenance, *table_rows])
    quality_pass = (
        annotated_success_rate == 1.0
        and min(generated_f1, default=0.0) >= 0.98
        and min(generated_order, default=0.0) >= 0.97
        and structure_score >= 0.95
        and exact_generated_structure
        and public_success_rate >= 0.99
        and public_success_rate + 0.01 >= public_reference_success_rate
        and crashes_or_timeouts == 0
    )
    return QualitySummary(
        candidate=candidate,
        annotated_success_rate=annotated_success_rate,
        annotated_min_ngram_f1=min(generated_f1, default=0.0),
        annotated_min_anchor_order=min(generated_order, default=0.0),
        segment_boundary_accuracy=boundary_score,
        provenance_accuracy=provenance_score,
        table_row_accuracy=table_score,
        public_success_rate=public_success_rate,
        public_reference_success_rate=public_reference_success_rate,
        public_mean_ngram_f1=_mean(public_f1),
        private_samples=private_total,
        private_success_rate=private_success_rate,
        crashes_or_timeouts=crashes_or_timeouts,
        quality_score=quality_score,
        quality_pass=quality_pass,
        cases=tuple(cases),
    )


def _quality_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# DOCX extractor quality report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Annotated fixtures: `{report['sample_counts']['annotated']}`",
        f"- Public corpus: `{report['sample_counts']['public']}`",
        f"- Private corpus: `{report['sample_counts']['private']}`",
        f"- Public source revision: `{report['public_corpus']['revision']}`",
        "",
        "## Summary",
        "",
        "| Candidate | Annotated success | Min F1 | Min order | Boundaries | Provenance | "
        "Table rows | Public success | Public mean F1 | Score | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['candidate']} | {summary['annotated_success_rate']:.3f} | "
            f"{summary['annotated_min_ngram_f1']:.3f} | {summary['annotated_min_anchor_order']:.3f} | "
            f"{summary['segment_boundary_accuracy']:.3f} | {summary['provenance_accuracy']:.3f} | "
            f"{summary['table_row_accuracy']:.3f} | {summary['public_success_rate']:.3f} | "
            f"{summary['public_mean_ngram_f1']:.3f} | {summary['quality_score']:.2f} | "
            f"{'PASS' if summary['quality_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Annotated fixtures", ""])
    for summary in report["summaries"]:
        lines.extend(
            [
                f"### {summary['candidate']}",
                "",
                "| Fixture | Status | F1 | Order | Boundaries | Provenance | Table rows |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in summary["cases"]:
            if case["source"] != "annotated" or case["sample_id"] == "corrupt":
                continue
            lines.append(
                f"| {case['sample_id']} | {case['actual_status']} | {_optional_score(case['ngram_f1'])} | "
                f"{_optional_score(case['anchor_order'])} | {_optional_score(case['segment_boundaries'])} | "
                f"{_optional_score(case['provenance'])} | {_optional_score(case['table_rows'])} |"
            )
        lines.append("")
    lines.extend(["## Lowest public-corpus comparisons", ""])
    for summary in report["summaries"]:
        public_cases = [case for case in summary["cases"] if case["source"] == "public"]
        public_cases.sort(key=lambda case: -1.0 if case["ngram_f1"] is None else case["ngram_f1"])
        lines.extend([f"### {summary['candidate']}", "", "| Sample | Status | F1 | Edit | Order | Error |"])
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for case in public_cases[:10]:
            error = (case["error"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {case['sample_id']} | {case['actual_status']} | {_optional_score(case['ngram_f1'])} | "
                f"{_optional_score(case['edit_similarity'])} | {_optional_score(case['anchor_order'])} | "
                f"{error} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate",
            "",
            "A candidate passes only with exact required structure on generated fixtures, annotated 3-gram F1 "
            "at least 0.98, order at least 0.97, public valid-file success at least 99%, no more than one "
            "percentage point below Python, and zero crashes or timeouts.",
            "",
            "Per-sample metrics and bounded diffs are available in `report.json`. Private paths and text are omitted.",
            "",
        ]
    )
    return "\n".join(lines)


def run_docx_quality(*, public_corpus_dir: Path | None = None, private_corpus_dir: Path | None = None) -> Path:
    executable, build_seconds = _build_rust_runner()
    public_metadata, public_documents = load_docx_public_corpus(public_corpus_dir)
    private_documents = load_private_docx_corpus(private_corpus_dir)
    with tempfile.TemporaryDirectory(prefix="lode-docx-quality-") as temporary:
        generated = create_docx_quality_fixtures(Path(temporary))
        generated = [fixture for fixture in generated if fixture.valid]
        documents = [(fixture.fixture_id, fixture.path) for fixture in generated]
        documents.extend((document.sample_id, document.path) for document in [*public_documents, *private_documents])
        results = _run_all_candidates(executable, documents)
        summaries = [
            _score_quality_candidate(
                candidate,
                generated,
                [*public_documents, *private_documents],
                results,
            )
            for candidate in ("python", *DOCX_RUST_CANDIDATES)
        ]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-docx-quality")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "docx",
        "round": "quality",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
        },
        "build_seconds": build_seconds,
        "sample_counts": {
            "annotated": len(generated),
            "public": len(public_documents),
            "private": len(private_documents),
        },
        "public_corpus": asdict(public_metadata),
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _quality_markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def _score_pdf_smoke_candidate(
    candidate: str,
    results: list[tuple[Fixture, ExtractionResult]],
    license_name: str,
    *,
    dependency_licenses_allowed: bool,
) -> PdfSmokeSummary:
    cases: list[PdfSmokeCase] = []
    valid_parsed = 0
    invalid_rejected = 0
    anchor_scores: list[float] = []
    boundary_scores: list[float] = []
    provenance_scores: list[float] = []
    markdown_expected = 0
    markdown_emitted = 0
    markdown_balance: list[bool] = []
    process_ok = True
    for fixture, result in results:
        process_ok = process_ok and result.process_status == "ok"
        if fixture.valid and result.status == "ok":
            valid_parsed += 1
            anchors = anchor_order_accuracy(result.text, fixture.anchors)
            expected_segments = _expected_segment_tuples(fixture)
            actual_segments = _segment_tuples(result)
            boundaries = segment_boundary_accuracy(actual_segments, expected_segments)
            provenance = segment_provenance_accuracy(actual_segments, expected_segments)
            if fixture.selection_scope:
                anchor_scores.append(anchors)
                boundary_scores.append(boundaries)
                provenance_scores.append(provenance)
            if fixture.expected_text:
                markdown_expected += 1
                if result.markdown:
                    markdown_emitted += 1
                    markdown_balance.append(markdown_delimiters_balanced(result.markdown))
            emitted = bool(result.markdown)
            balanced = markdown_delimiters_balanced(result.markdown) if result.markdown else None
        else:
            anchors = boundaries = provenance = None
            emitted = False
            balanced = None
        if not fixture.valid and result.status == "error":
            invalid_rejected += 1
        cases.append(
            PdfSmokeCase(
                fixture_id=fixture.fixture_id,
                selection_scope=fixture.selection_scope,
                expected_status="ok" if fixture.valid else "error",
                actual_status=result.status,
                anchor_order=anchors,
                segment_boundaries=boundaries,
                provenance=provenance,
                markdown_emitted=emitted,
                markdown_delimiters_balanced=balanced,
                process_status=result.process_status,
                error=result.error,
            )
        )
    valid_total = sum(fixture.valid for fixture, _ in results)
    invalid_total = len(results) - valid_total
    direct_license_allowed = candidate == "python" or license_name in ALLOWED_LICENSES
    license_allowed = direct_license_allowed and (candidate == "python" or dependency_licenses_allowed)
    smoke_pass = process_ok and license_allowed and valid_parsed == valid_total and invalid_rejected == invalid_total
    return PdfSmokeSummary(
        candidate=candidate,
        license=license_name,
        license_allowed=license_allowed,
        valid_parsed=valid_parsed,
        valid_total=valid_total,
        invalid_rejected=invalid_rejected,
        invalid_total=invalid_total,
        minimum_anchor_order=min(anchor_scores, default=0.0),
        boundary_accuracy=_mean(boundary_scores),
        provenance_accuracy=_mean(provenance_scores),
        markdown_coverage=markdown_emitted / markdown_expected if markdown_expected else 0.0,
        markdown_delimiters_balanced=all(markdown_balance) if markdown_balance else None,
        smoke_pass=smoke_pass,
        cases=tuple(cases),
    )


def _pdf_smoke_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# PDF extractor smoke report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Rust runner build: `{report['build_seconds']:.2f}s`",
        "",
        "## Summary",
        "",
        "| Candidate | License | Valid | Invalid rejected | Min order | Boundaries | Provenance | "
        "Markdown coverage | Delimiter parity | Smoke |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in report["summaries"]:
        parity = summary["markdown_delimiters_balanced"]
        parity_text = "N/A" if parity is None else ("PASS" if parity else "FAIL")
        lines.append(
            f"| {summary['candidate']} | {summary['license']} | "
            f"{summary['valid_parsed']}/{summary['valid_total']} | "
            f"{summary['invalid_rejected']}/{summary['invalid_total']} | "
            f"{summary['minimum_anchor_order']:.3f} | {summary['boundary_accuracy']:.3f} | "
            f"{summary['provenance_accuracy']:.3f} | {summary['markdown_coverage']:.3f} | "
            f"{parity_text} | {'PASS' if summary['smoke_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Cases", ""])
    for summary in report["summaries"]:
        lines.extend(
            [
                f"### {summary['candidate']}",
                "",
                "| Fixture | Scope | Expected | Actual | Order | Boundaries | Provenance | Markdown | Parity | Error |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for case in summary["cases"]:
            parity = case["markdown_delimiters_balanced"]
            parity_text = "N/A" if parity is None else ("PASS" if parity else "FAIL")
            error = (case["error"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {case['fixture_id']} | {'selection' if case['selection_scope'] else 'diagnostic'} | "
                f"{case['expected_status']} | {case['actual_status']} | "
                f"{_optional_score(case['anchor_order'])} | {_optional_score(case['segment_boundaries'])} | "
                f"{_optional_score(case['provenance'])} | {'yes' if case['markdown_emitted'] else 'no'} | "
                f"{parity_text} | {error} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Markdown interpretation",
            "",
            "Markdown coverage means that the adapter returned a non-empty Markdown representation for every "
            "text-bearing generated page. Delimiter parity checks only fenced code, inline code, strong, "
            "underscore emphasis, and strikethrough token counts; it is a smoke signal for symmetric token "
            "completion, not a Markdown parser correctness proof.",
            "",
            "The current Lode Segment contract stores plain text plus heading/page provenance. A candidate's "
            "Markdown output is therefore an optional capability and is not yet preserved by ingestion.",
            "",
            "## Gate",
            "",
            "Smoke requires allowed licenses, successful parsing of every valid fixture, explicit rejection of "
            "encrypted and corrupt inputs, and no crash or timeout. Reading order, structure, provenance, and "
            "Markdown columns are diagnostic in this round. Quality promotes only supported selection-scope "
            "metrics; multi-column and table reconstruction remain non-gating known limitations.",
            "",
            "`pdfium-render` is not admitted to this round because the required three-platform static Pdfium "
            "artifact setup has not been demonstrated.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pdf_smoke() -> Path:
    executable, build_seconds = _build_rust_runner()
    metadata = _candidate_metadata()
    with tempfile.TemporaryDirectory(prefix="lode-pdf-smoke-") as temporary:
        fixtures = create_pdf_smoke_fixtures(Path(temporary))
        all_results: dict[str, list[tuple[Fixture, ExtractionResult]]] = {
            "python": [(fixture, _python_extract(fixture.path)) for fixture in fixtures]
        }
        for candidate in PDF_RUST_CANDIDATES:
            all_results[candidate] = [
                (fixture, _rust_extract(executable, candidate, fixture.path)) for fixture in fixtures
            ]

    dependency_licenses_allowed, license_output = _license_audit()
    package_names = {"pdf_oxide": "pdf_oxide", "pdf_extract": "pdf-extract"}
    versions = {"python": "PyMuPDF 1.28.2"}
    summaries = [
        _score_pdf_smoke_candidate(
            "python",
            all_results["python"],
            "baseline",
            dependency_licenses_allowed=dependency_licenses_allowed,
        )
    ]
    for candidate in PDF_RUST_CANDIDATES:
        version, license_name = metadata[package_names[candidate]]
        versions[candidate] = version
        summaries.append(
            _score_pdf_smoke_candidate(
                candidate,
                all_results[candidate],
                license_name,
                dependency_licenses_allowed=dependency_licenses_allowed,
            )
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-pdf-smoke")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "pdf",
        "round": "smoke",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
        },
        "versions": versions,
        "build_seconds": build_seconds,
        "license_audit": {"passed": dependency_licenses_allowed, "output": license_output},
        "summaries": [asdict(summary) for summary in summaries],
        "results": {
            candidate: [
                {"fixture_id": fixture.fixture_id, "valid": fixture.valid, "result": asdict(result)}
                for fixture, result in results
            ]
            for candidate, results in all_results.items()
        },
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _pdf_smoke_markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def _pdf_markdown_scores(
    result: ExtractionResult,
    expected_text: str,
    anchors: tuple[str, ...],
    expected_headings: tuple[tuple[int, str], ...],
) -> tuple[float | None, float | None, float | None, float | None, bool | None]:
    if not result.markdown or not expected_text:
        return None, None, None, None, None
    plain = markdown_plain_text(result.markdown)
    precision, _, f1 = ngram_prf(plain, expected_text)
    return (
        precision,
        f1,
        anchor_order_accuracy(plain, anchors),
        markdown_heading_accuracy(result.markdown, expected_headings),
        markdown_delimiters_balanced(result.markdown),
    )


def _score_pdf_quality_candidate(
    candidate: str,
    generated: list[Fixture],
    corpus_documents: list[CorpusDocument],
    results: dict[str, dict[str, ExtractionResult]],
) -> PdfQualitySummary:
    cases: list[PdfQualityCase] = []
    generated_f1: list[float] = []
    generated_order: list[float] = []
    boundaries: list[float] = []
    provenance: list[float] = []
    public_f1: list[float] = []
    markdown_precision: list[float] = []
    markdown_f1: list[float] = []
    markdown_order: list[float] = []
    markdown_headings: list[float] = []
    markdown_balance: list[bool] = []
    markdown_expected = 0
    markdown_emitted = 0
    annotated_success = 0
    public_success = 0
    public_reference_success = 0
    private_success = 0
    private_total = 0
    crashes_or_timeouts = 0

    for fixture in generated:
        result = results[candidate][fixture.fixture_id]
        expected_status = "ok" if fixture.valid else "error"
        if result.process_status != "ok":
            crashes_or_timeouts += 1
        if result.status == expected_status:
            annotated_success += 1
        if fixture.valid and result.status == "ok":
            _, _, f1 = ngram_prf(result.text, fixture.expected_text)
            edit = normalized_edit_similarity(result.text, fixture.expected_text)
            order = anchor_order_accuracy(result.text, fixture.anchors)
            expected_segments = _expected_segment_tuples(fixture)
            actual_segments = _segment_tuples(result)
            boundary = segment_content_boundary_accuracy(actual_segments, expected_segments)
            provenance_score = segment_provenance_accuracy(actual_segments, expected_segments)
            if fixture.selection_scope:
                generated_f1.append(f1)
                generated_order.append(order)
                boundaries.append(boundary)
                provenance.append(provenance_score)
                if fixture.expected_text:
                    markdown_expected += 1
            md_precision, md_f1, md_order, md_headings, md_balanced = _pdf_markdown_scores(
                result,
                fixture.expected_text,
                fixture.anchors,
                fixture.expected_markdown_headings,
            )
            if fixture.selection_scope and md_f1 is not None:
                markdown_emitted += 1
                markdown_precision.append(md_precision or 0.0)
                markdown_f1.append(md_f1)
                markdown_order.append(md_order or 0.0)
                markdown_headings.append(md_headings or 0.0)
                markdown_balance.append(bool(md_balanced))
        else:
            f1 = edit = order = boundary = provenance_score = None
            md_precision = md_f1 = md_order = md_headings = md_balanced = None
        cases.append(
            PdfQualityCase(
                sample_id=fixture.fixture_id,
                source="annotated",
                selection_scope=fixture.selection_scope,
                reference_status=expected_status,
                actual_status=result.status,
                ngram_f1=f1,
                edit_similarity=edit,
                anchor_order=order,
                segment_boundaries=boundary,
                provenance=provenance_score,
                markdown_precision=md_precision,
                markdown_f1=md_f1,
                markdown_order=md_order,
                markdown_headings=md_headings,
                markdown_balanced=md_balanced,
                process_status=result.process_status,
                error=result.error,
                diff_excerpt=_diff_excerpt(result.text, fixture.expected_text, source="annotated"),
            )
        )

    for document in corpus_documents:
        result = results[candidate][document.sample_id]
        reference = results["python"][document.sample_id]
        if result.process_status != "ok":
            crashes_or_timeouts += 1
        if document.source == "public":
            public_success += result.status == "ok"
            public_reference_success += reference.status == "ok"
        else:
            private_total += 1
            private_success += result.status == "ok"
        anchors = _reference_anchors(reference.text) if reference.status == "ok" else ()
        if result.status == "ok" and reference.status == "ok":
            _, _, f1 = ngram_prf(result.text, reference.text)
            edit = normalized_edit_similarity(result.text, reference.text)
            order = anchor_order_accuracy(result.text, anchors)
            if document.source == "public":
                public_f1.append(f1)
            if reference.text:
                markdown_expected += 1
            md_precision, md_f1, md_order, _, md_balanced = _pdf_markdown_scores(result, reference.text, anchors, ())
            md_headings = None
            if md_f1 is not None:
                markdown_emitted += 1
                markdown_precision.append(md_precision or 0.0)
                markdown_f1.append(md_f1)
                markdown_order.append(md_order or 0.0)
                markdown_balance.append(bool(md_balanced))
        else:
            f1 = edit = order = None
            md_precision = md_f1 = md_order = md_headings = md_balanced = None
        cases.append(
            PdfQualityCase(
                sample_id=document.sample_id,
                source=document.source,
                selection_scope=True,
                reference_status=reference.status,
                actual_status=result.status,
                ngram_f1=f1,
                edit_similarity=edit,
                anchor_order=order,
                segment_boundaries=None,
                provenance=None,
                markdown_precision=md_precision,
                markdown_f1=md_f1,
                markdown_order=md_order,
                markdown_headings=md_headings,
                markdown_balanced=md_balanced,
                process_status=result.process_status,
                error=result.error,
                diff_excerpt=(
                    _diff_excerpt(result.text, reference.text, source=document.source)
                    if result.status == "ok" and reference.status == "ok"
                    else None
                ),
            )
        )

    annotated_success_rate = annotated_success / len(generated) if generated else 0.0
    public_total = sum(document.source == "public" for document in corpus_documents)
    public_success_rate = public_success / public_total if public_total else 0.0
    public_reference_success_rate = public_reference_success / public_total if public_total else 0.0
    private_success_rate = private_success / private_total if private_total else None
    boundary_score = _mean(boundaries)
    provenance_score = _mean(provenance)
    content_score = _mean([_mean(generated_f1), _mean(public_f1)])
    order_score = _mean(generated_order)
    robustness_score = _mean([annotated_success_rate, public_success_rate])
    quality_score = 100 * (
        0.40 * content_score
        + 0.25 * order_score
        + 0.25 * _mean([boundary_score, provenance_score])
        + 0.10 * robustness_score
    )
    quality_pass = (
        annotated_success_rate == 1.0
        and min(generated_f1, default=0.0) >= 0.98
        and min(generated_order, default=0.0) >= 0.97
        and boundary_score >= 0.95
        and provenance_score >= 0.95
        and public_success_rate >= 0.99
        and public_success_rate + 0.01 >= public_reference_success_rate
        and crashes_or_timeouts == 0
    )
    return PdfQualitySummary(
        candidate=candidate,
        annotated_success_rate=annotated_success_rate,
        annotated_min_ngram_f1=min(generated_f1, default=0.0),
        annotated_min_anchor_order=min(generated_order, default=0.0),
        segment_boundary_accuracy=boundary_score,
        provenance_accuracy=provenance_score,
        public_success_rate=public_success_rate,
        public_reference_success_rate=public_reference_success_rate,
        public_mean_ngram_f1=_mean(public_f1),
        markdown_coverage=markdown_emitted / markdown_expected if markdown_expected else 0.0,
        markdown_mean_precision=_mean(markdown_precision) if markdown_precision else None,
        markdown_mean_f1=_mean(markdown_f1) if markdown_f1 else None,
        markdown_min_order=min(markdown_order) if markdown_order else None,
        markdown_heading_accuracy=_mean(markdown_headings) if markdown_headings else None,
        markdown_balanced=all(markdown_balance) if markdown_balance else None,
        private_samples=private_total,
        private_success_rate=private_success_rate,
        crashes_or_timeouts=crashes_or_timeouts,
        quality_score=quality_score,
        quality_pass=quality_pass,
        cases=tuple(cases),
    )


def _pdf_quality_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# PDF extractor quality report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Annotated fixtures: `{report['sample_counts']['annotated']}`",
        f"- Public corpus: `{report['sample_counts']['public']}`",
        f"- Private corpus: `{report['sample_counts']['private']}`",
        "",
        "## Plain-text selection summary",
        "",
        "| Candidate | Annotated success | Min F1 | Min order | Boundaries | Provenance | "
        "Public success | Public mean F1 | Score | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['candidate']} | {summary['annotated_success_rate']:.3f} | "
            f"{summary['annotated_min_ngram_f1']:.3f} | {summary['annotated_min_anchor_order']:.3f} | "
            f"{summary['segment_boundary_accuracy']:.3f} | {summary['provenance_accuracy']:.3f} | "
            f"{summary['public_success_rate']:.3f} | {summary['public_mean_ngram_f1']:.3f} | "
            f"{summary['quality_score']:.2f} | {'PASS' if summary['quality_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Markdown diagnostics",
            "",
            "| Candidate | Coverage | Mean precision | Mean F1 | Min order | Heading accuracy | Balanced |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for summary in report["summaries"]:
        balanced = summary["markdown_balanced"]
        lines.append(
            f"| {summary['candidate']} | {summary['markdown_coverage']:.3f} | "
            f"{_optional_score(summary['markdown_mean_precision'])} | "
            f"{_optional_score(summary['markdown_mean_f1'])} | "
            f"{_optional_score(summary['markdown_min_order'])} | "
            f"{_optional_score(summary['markdown_heading_accuracy'])} | "
            f"{'N/A' if balanced is None else ('PASS' if balanced else 'FAIL')} |"
        )
    lines.extend(["", "## Annotated fixtures", ""])
    for summary in report["summaries"]:
        lines.extend(
            [
                f"### {summary['candidate']}",
                "",
                "| Fixture | Scope | Status | F1 | Order | Boundaries | Provenance | MD F1 | MD order | MD headings |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in summary["cases"]:
            if case["source"] != "annotated":
                continue
            lines.append(
                f"| {case['sample_id']} | {'selection' if case['selection_scope'] else 'diagnostic'} | "
                f"{case['actual_status']} | {_optional_score(case['ngram_f1'])} | "
                f"{_optional_score(case['anchor_order'])} | {_optional_score(case['segment_boundaries'])} | "
                f"{_optional_score(case['provenance'])} | {_optional_score(case['markdown_f1'])} | "
                f"{_optional_score(case['markdown_order'])} | {_optional_score(case['markdown_headings'])} |"
            )
        lines.append("")
    lines.extend(["## Lowest public-corpus comparisons", ""])
    for summary in report["summaries"]:
        public_cases = [case for case in summary["cases"] if case["source"] == "public"]
        public_cases.sort(key=lambda case: -1.0 if case["ngram_f1"] is None else case["ngram_f1"])
        lines.extend([f"### {summary['candidate']}", "", "| Sample | Status | F1 | Edit | Order | MD F1 | Error |"])
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for case in public_cases[:10]:
            error = (case["error"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {case['sample_id']} | {case['actual_status']} | {_optional_score(case['ngram_f1'])} | "
                f"{_optional_score(case['edit_similarity'])} | {_optional_score(case['anchor_order'])} | "
                f"{_optional_score(case['markdown_f1'])} | {error} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate and scope",
            "",
            "The production gate uses supported single-flow plain text only: all annotated status expectations, "
            "minimum annotated F1 0.98, order 0.97, boundary/provenance 0.95, public success 99%, no more than "
            "one percentage point below Python, and zero crashes/timeouts. Markdown is diagnostic. Multi-column "
            "and PDF table reconstruction are non-gating known limitations.",
            "",
            "Per-sample metrics and bounded diffs are available in `report.json`. Private paths and text are omitted.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pdf_quality(*, public_corpus_dir: Path | None = None, private_corpus_dir: Path | None = None) -> Path:
    executable, build_seconds = _build_rust_runner()
    public_metadata, public_documents = load_pdf_public_corpus(public_corpus_dir)
    private_documents = load_private_pdf_corpus(private_corpus_dir)
    with tempfile.TemporaryDirectory(prefix="lode-pdf-quality-") as temporary:
        generated = create_pdf_quality_fixtures(Path(temporary))
        documents = [(fixture.fixture_id, fixture.path) for fixture in generated]
        documents.extend((document.sample_id, document.path) for document in [*public_documents, *private_documents])
        results: dict[str, dict[str, ExtractionResult]] = {
            "python": {sample_id: _python_extract(path) for sample_id, path in documents},
            "pdf_oxide": {sample_id: _rust_extract(executable, "pdf_oxide", path) for sample_id, path in documents},
        }
        summaries = [
            _score_pdf_quality_candidate(
                candidate,
                generated,
                [*public_documents, *private_documents],
                results,
            )
            for candidate in ("python", "pdf_oxide")
        ]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-pdf-quality")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "pdf",
        "round": "quality",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
        },
        "versions": {"python": "PyMuPDF 1.28.2", "pdf_oxide": "0.3.78"},
        "build_seconds": build_seconds,
        "sample_counts": {
            "annotated": len(generated),
            "public": len(public_documents),
            "private": len(private_documents),
        },
        "public_corpora": [asdict(metadata) for metadata in public_metadata],
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _pdf_quality_markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def _compatibility_text(text: str) -> str:
    """Normalize compatibility glyphs without changing the round-1 NFC contract."""
    return unicodedata.normalize("NFKC", text)


def _page_anchor_accuracy(
    result: ExtractionResult,
    expected: tuple[tuple[str, ...], ...],
    *,
    compatibility: bool,
) -> float:
    if not expected:
        return 1.0
    segments = {segment.page: segment.text for segment in result.segments}
    scores: list[float] = []
    for page, anchors in enumerate(expected, start=1):
        text = segments.get(page, "")
        if compatibility:
            text = _compatibility_text(text)
            anchors = tuple(_compatibility_text(anchor) for anchor in anchors)
        scores.append(anchor_order_accuracy(text, anchors))
    return _mean(scores)


def _score_pdf_adjudication_candidate(
    candidate: str,
    truths: tuple[ManualPdfTruth, ...],
    results: dict[str, ExtractionResult],
) -> PdfAdjudicationSummary:
    cases: list[PdfAdjudicationCase] = []
    strict_f1_scores: list[float] = []
    compatibility_f1_scores: list[float] = []
    selection_passed = 0
    diagnostic_passed = 0
    for truth in truths:
        result = results[truth.sample_id]
        strict_anchors = anchor_order_accuracy(result.text, truth.anchors) if result.status == "ok" else 0.0
        compatibility_anchors = (
            anchor_order_accuracy(
                _compatibility_text(result.text),
                tuple(_compatibility_text(anchor) for anchor in truth.anchors),
            )
            if result.status == "ok"
            else 0.0
        )
        strict_pages = (
            _page_anchor_accuracy(result, truth.page_anchors, compatibility=False) if result.status == "ok" else 0.0
        )
        compatibility_pages = (
            _page_anchor_accuracy(result, truth.page_anchors, compatibility=True) if result.status == "ok" else 0.0
        )
        if truth.expected_text is not None and result.status == "ok":
            _, _, strict_f1 = ngram_prf(result.text, truth.expected_text)
            _, _, compatibility_f1 = ngram_prf(
                _compatibility_text(result.text),
                _compatibility_text(truth.expected_text),
            )
            strict_f1_scores.append(strict_f1)
            compatibility_f1_scores.append(compatibility_f1)
        else:
            strict_f1 = compatibility_f1 = None
        invalid_unicode_characters = invalid_unicode_character_count(result.text)
        control_characters = sum(
            unicodedata.category(character) == "Cc" and character not in "\n\r\t" for character in result.text
        )
        if result.markdown:
            markdown_anchors = anchor_order_accuracy(result.markdown, truth.anchors)
            markdown_balanced = markdown_delimiters_balanced(result.markdown)
        else:
            markdown_anchors = None
            markdown_balanced = None
        case_pass = (
            result.status == "ok"
            and (strict_f1 is None or strict_f1 >= 0.98)
            and strict_anchors >= 0.97
            and strict_pages >= 0.97
            and invalid_unicode_characters == 0
            and control_characters == 0
        )
        if case_pass:
            if truth.selection_scope:
                selection_passed += 1
            else:
                diagnostic_passed += 1
        cases.append(
            PdfAdjudicationCase(
                sample_id=truth.sample_id,
                category=truth.category,
                selection_scope=truth.selection_scope,
                actual_status=result.status,
                strict_ngram_f1=strict_f1,
                compatibility_ngram_f1=compatibility_f1,
                strict_anchor_order=strict_anchors,
                compatibility_anchor_order=compatibility_anchors,
                strict_page_anchor_accuracy=strict_pages,
                compatibility_page_anchor_accuracy=compatibility_pages,
                invalid_unicode_characters=invalid_unicode_characters,
                control_characters=control_characters,
                markdown_anchor_order=markdown_anchors,
                markdown_balanced=markdown_balanced,
                case_pass=case_pass,
                note=truth.note,
                error=result.error,
            )
        )
    selection_total = sum(truth.selection_scope for truth in truths)
    diagnostic_total = len(truths) - selection_total
    return PdfAdjudicationSummary(
        candidate=candidate,
        selection_passed=selection_passed,
        selection_total=selection_total,
        diagnostic_passed=diagnostic_passed,
        diagnostic_total=diagnostic_total,
        strict_mean_ngram_f1=_mean(strict_f1_scores),
        compatibility_mean_ngram_f1=_mean(compatibility_f1_scores),
        focused_quality_pass=selection_passed == selection_total,
        cases=tuple(cases),
    )


def _pdf_adjudication_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# PDF extractor focused quality report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Manually adjudicated samples: `{report['sample_count']}`",
        "",
        "## Summary",
        "",
        "| Candidate | Selection pass | Diagnostic pass | Strict mean F1 | Compatibility mean F1 | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['candidate']} | {summary['selection_passed']}/{summary['selection_total']} | "
            f"{summary['diagnostic_passed']}/{summary['diagnostic_total']} | "
            f"{summary['strict_mean_ngram_f1']:.3f} | {summary['compatibility_mean_ngram_f1']:.3f} | "
            f"{'PASS' if summary['focused_quality_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Cases", ""])
    for summary in report["summaries"]:
        lines.extend(
            [
                f"### {summary['candidate']}",
                "",
                "| Sample | Scope | Status | NFC F1 | NFKC F1 | NFC order | NFKC order | "
                "Page anchors | NFKC page anchors | Invalid Unicode | Controls | Markdown order | Result |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for case in summary["cases"]:
            lines.append(
                f"| {case['sample_id']} | {'selection' if case['selection_scope'] else 'diagnostic'} | "
                f"{case['actual_status']} | {_optional_score(case['strict_ngram_f1'])} | "
                f"{_optional_score(case['compatibility_ngram_f1'])} | "
                f"{case['strict_anchor_order']:.3f} | {case['compatibility_anchor_order']:.3f} | "
                f"{case['strict_page_anchor_accuracy']:.3f} | "
                f"{case['compatibility_page_anchor_accuracy']:.3f} | "
                f"{case['invalid_unicode_characters']} | {case['control_characters']} | "
                f"{_optional_score(case['markdown_anchor_order'])} | "
                f"{'PASS' if case['case_pass'] else 'FAIL'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The gate continues to use the planned NFC text contract. NFKC is reported separately to show when "
            "a disagreement is caused by Arabic presentation forms or other compatibility glyphs; it does not "
            "silently relax the production contract. A selection-scope case requires status OK, NFC F1 0.98 "
            "when full truth is available, anchor and page-anchor accuracy 0.97, and no replacement or control "
            "characters.",
            "",
            "The malformed ToUnicode formula and 14-page character grid remain diagnostic because safe formula "
            "linearization and table-like grid reconstruction are outside the selected PDF scope. Markdown is "
            "diagnostic and is never substituted for canonical page text.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pdf_quality_round_2(*, public_corpus_dir: Path | None = None) -> Path:
    executable, build_seconds = _build_rust_runner()
    public_metadata, public_documents = load_pdf_public_corpus(public_corpus_dir)
    documents_by_id = {document.sample_id: document for document in public_documents}
    truths = load_pdf_quality_round_2_truth()
    for truth in truths:
        document = documents_by_id.get(truth.sample_id)
        if document is None:
            raise ValueError(f"manual truth references unknown sample: {truth.sample_id}")
        verify_manual_truth_document(truth, document.path)
    results = {
        "python": {truth.sample_id: _python_extract(documents_by_id[truth.sample_id].path) for truth in truths},
        "pdf_oxide": {
            truth.sample_id: _rust_extract(
                executable,
                "pdf_oxide",
                documents_by_id[truth.sample_id].path,
            )
            for truth in truths
        },
        "pdf_oxide_remediated": {
            truth.sample_id: _rust_extract(
                executable,
                "pdf_oxide_remediated",
                documents_by_id[truth.sample_id].path,
            )
            for truth in truths
        },
    }
    summaries = [
        _score_pdf_adjudication_candidate(candidate, truths, results[candidate])
        for candidate in ("python", "pdf_oxide", "pdf_oxide_remediated")
    ]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-pdf-quality-2")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": "pdf",
        "round": "quality-2",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
        },
        "versions": {
            "python": "PyMuPDF 1.28.2",
            "pdf_oxide": "0.3.78",
            "pdf_oxide_remediated": "0.3.78 + runner NFKC/landscape-line prototype",
        },
        "build_seconds": build_seconds,
        "sample_count": len(truths),
        "public_corpora": [asdict(metadata) for metadata in public_metadata],
        "truth": [asdict(truth) for truth in truths],
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _pdf_adjudication_markdown_report(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Report: {output_dir / 'report.md'}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged document extractor evaluations")
    parser.add_argument("--format", choices=("doc", "docx", "pdf"), required=True)
    parser.add_argument("--round", choices=("smoke", "quality", "quality-markdown", "quality-2"), required=True)
    parser.add_argument("--public-corpus-dir", type=Path)
    parser.add_argument("--private-corpus-dir", type=Path)
    args = parser.parse_args()
    if (args.format, args.round) == ("doc", "smoke"):
        run_doc_smoke(public_corpus_dir=args.public_corpus_dir)
    elif (args.format, args.round) == ("doc", "quality"):
        run_doc_quality(public_corpus_dir=args.public_corpus_dir)
    elif (args.format, args.round) == ("docx", "smoke"):
        run_docx_smoke()
    elif (args.format, args.round) == ("docx", "quality"):
        run_docx_quality(
            public_corpus_dir=args.public_corpus_dir,
            private_corpus_dir=args.private_corpus_dir,
        )
    elif (args.format, args.round) == ("docx", "quality-markdown"):
        from evals.document_extractors.markdown_quality import run_docx_markdown_quality

        run_docx_markdown_quality(
            public_corpus_dir=args.public_corpus_dir,
            private_corpus_dir=args.private_corpus_dir,
        )
    elif (args.format, args.round) == ("pdf", "smoke"):
        run_pdf_smoke()
    elif (args.format, args.round) == ("pdf", "quality"):
        run_pdf_quality(
            public_corpus_dir=args.public_corpus_dir,
            private_corpus_dir=args.private_corpus_dir,
        )
    elif (args.format, args.round) == ("pdf", "quality-2"):
        run_pdf_quality_round_2(public_corpus_dir=args.public_corpus_dir)
    else:
        parser.error(f"round {args.round!r} is not implemented for {args.format!r}")


if __name__ == "__main__":
    main()
