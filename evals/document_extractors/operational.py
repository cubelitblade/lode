"""Operational measurements for the selected Rust document extractors.

This round deliberately measures the selected candidate only.  It is a
production-readiness report, not another quality-selection experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from evals.document_extractors.fixtures import (
    Fixture,
    create_docx_quality_fixtures,
    create_pdf_quality_fixtures,
)
from evals.document_extractors.run import (
    ROOT,
    RUST_TOOLCHAIN,
    _cargo_command,  # pyright: ignore[reportPrivateUsage] - preserve the selected toolchain.
    _rustc_version,  # pyright: ignore[reportPrivateUsage] - include compiler provenance in the report.
)

TIMEOUT_SECONDS = 15
ITERATIONS = 3
PRODUCTION_MANIFEST = ROOT / "evals" / "document_extractors" / "production-runner" / "Cargo.toml"
PRODUCTION_TARGET_DIR = Path(tempfile.gettempdir()) / "lode-production-extractor-target"


def _timed_extract(executable: Path, fixture: Fixture) -> dict[str, Any]:
    """Run one fixture and capture timing, status, and best-effort RSS."""
    command = [str(executable), fixture.path.suffix.lower().lstrip("."), str(fixture.path)]
    timed_command = command
    has_time = os.name != "nt" and Path("/usr/bin/time").exists()
    if has_time:
        timed_command = ["/usr/bin/time", "-f", "%M", *command]
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            timed_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "process_status": "timeout",
            "elapsed_ns": 0,
            "rss_peak_kib": None,
            "error": f"candidate timed out after {TIMEOUT_SECONDS} seconds",
        }
    elapsed_ns = time.perf_counter_ns() - started
    stderr_lines = completed.stderr.splitlines()
    rss_peak_kib: int | None = None
    if has_time and stderr_lines and stderr_lines[-1].strip().isdigit():
        rss_peak_kib = int(stderr_lines[-1].strip())
        stderr_lines = stderr_lines[:-1]
    if completed.returncode != 0:
        detail = "\n".join(stderr_lines).strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return {
            "status": "error",
            "process_status": "crash",
            "elapsed_ns": elapsed_ns,
            "rss_peak_kib": rss_peak_kib,
            "error": detail,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "process_status": "protocol_error",
            "elapsed_ns": elapsed_ns,
            "rss_peak_kib": rss_peak_kib,
            "error": f"invalid runner output: {exc}",
        }
    return {
        "status": str(payload.get("status", "error")),
        "process_status": "ok",
        "elapsed_ns": int(payload.get("elapsed_ns", elapsed_ns)),
        "rss_peak_kib": rss_peak_kib,
        "text": str(payload.get("text", "")),
        "segments": payload.get("segments", []),
        "error": None if payload.get("error") is None else str(payload["error"]),
    }


def _build_production_runner() -> tuple[Path, float]:
    started = time.perf_counter()
    environment = {**os.environ, "CARGO_TARGET_DIR": str(PRODUCTION_TARGET_DIR)}
    subprocess.run(
        _cargo_command("build", "--release", "--manifest-path", str(PRODUCTION_MANIFEST)),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    executable = PRODUCTION_TARGET_DIR / "release" / (
        "document-extractor-production-runner.exe"
        if os.name == "nt"
        else "document-extractor-production-runner"
    )
    return executable, time.perf_counter() - started


def _metadata() -> tuple[int, int]:
    environment = {**os.environ, "CARGO_TARGET_DIR": str(PRODUCTION_TARGET_DIR)}
    completed = subprocess.run(
        _cargo_command("metadata", "--format-version", "1", "--manifest-path", str(PRODUCTION_MANIFEST)),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    packages = payload.get("packages", [])
    dependency_count = sum(1 for package in packages if package.get("source"))
    executable = PRODUCTION_TARGET_DIR / "release" / (
        "document-extractor-production-runner.exe"
        if os.name == "nt"
        else "document-extractor-production-runner"
    )
    return dependency_count, executable.stat().st_size


def _production_binary_sizes() -> dict[str, int]:
    release = ROOT / "lode-rs" / "target" / "release"
    return {
        name: (release / (f"{name}.exe" if os.name == "nt" else name)).stat().st_size
        for name in ("lode", "lode-mcp")
        if (release / (f"{name}.exe" if os.name == "nt" else name)).is_file()
    }


def _build_production_binaries() -> tuple[float, dict[str, int]]:
    """Build the shipping workspace with the same pinned Rust toolchain."""
    started = time.perf_counter()
    subprocess.run(
        _cargo_command(
            "build",
            "--release",
            "--workspace",
            "--manifest-path",
            str(ROOT / "lode-rs" / "Cargo.toml"),
        ),
        cwd=ROOT,
        check=True,
    )
    return time.perf_counter() - started, _production_binary_sizes()


def _production_metadata() -> int:
    manifest = ROOT / "lode-rs" / "Cargo.toml"
    completed = subprocess.run(
        _cargo_command("metadata", "--format-version", "1", "--manifest-path", str(manifest)),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(completed.stdout).get("packages", [])
    dependency_count = sum(1 for package in packages if package.get("source"))
    return dependency_count


def _fixtures(format_name: str, directory: Path) -> list[Fixture]:
    fixtures = (
        create_docx_quality_fixtures(directory)
        if format_name == "docx"
        else create_pdf_quality_fixtures(directory)
    )
    # Two-column and other layout-reconstruction samples stay visible in the
    # report, while their valid/invalid status is excluded from the gate.
    return fixtures


def run_operational(format_name: str, *, iterations: int = ITERATIONS) -> Path:
    executable, build_seconds = _build_production_runner()
    # Use the production lode-core adapter rather than a standalone copy of
    # the selected library implementation.
    candidate = "lode_core"
    with tempfile.TemporaryDirectory(prefix=f"lode-{format_name}-operational-") as temporary:
        fixtures = _fixtures(format_name, Path(temporary))
        cases: list[dict[str, Any]] = []
        for fixture in fixtures:
            samples = [_timed_extract(executable, fixture) for _ in range(iterations)]
            timings = [int(sample["elapsed_ns"]) for sample in samples if sample["process_status"] == "ok"]
            rss_values = [int(sample["rss_peak_kib"]) for sample in samples if sample["rss_peak_kib"] is not None]
            statuses = [str(sample["status"]) for sample in samples]
            process_statuses = [str(sample["process_status"]) for sample in samples]
            expected_ok = fixture.valid
            in_gate = fixture.selection_scope or not fixture.valid
            expected_segments = [
                {"text": segment.text, "heading": segment.heading, "page": segment.page}
                for segment in fixture.expected_segments
            ]
            # The production adapter exposes the canonical document text as
            # segments joined by a blank line.  Compare that representation
            # rather than the legacy fixture shorthand, which predates the
            # Segment-only index contract.
            expected_text = "\n\n".join(segment.text for segment in fixture.expected_segments)
            truth_pass = all(
                not expected_ok
                or (sample.get("text") == expected_text and sample.get("segments") == expected_segments)
                for sample in samples
            )
            gate_ok = in_gate and all(
                status == ("ok" if expected_ok else "error") for status in statuses
            ) and all(status == "ok" for status in process_statuses) and truth_pass
            cases.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "valid": fixture.valid,
                    "selection_scope": fixture.selection_scope,
                    "iterations": iterations,
                    "input_bytes": fixture.path.stat().st_size,
                    "statuses": statuses,
                    "process_statuses": process_statuses,
                    "total_elapsed_ns": sum(timings),
                    "p50_ms": median(timings) / 1_000_000 if timings else None,
                    "p95_ms": sorted(timings)[max(0, math.ceil(len(timings) * 0.95) - 1)] / 1_000_000
                    if timings
                    else None,
                    "rss_peak_kib": max(rss_values) if rss_values else None,
                    "truth_pass": truth_pass,
                    "gate_pass": gate_ok,
                    "errors": [sample["error"] for sample in samples if sample["error"]],
                }
            )
    dependency_count, runner_binary_size = _metadata()
    production_build_seconds, production_binaries = _build_production_binaries()
    production_dependency_count = _production_metadata()
    total_elapsed_ns = sum(int(case["total_elapsed_ns"]) for case in cases)
    total_input_bytes = sum(int(case["input_bytes"]) * iterations for case in cases)
    run_id = datetime.now(UTC).strftime(f"%Y%m%dT%H%M%SZ-{format_name}-operational")
    report: dict[str, Any] = {
        "run_id": run_id,
        "format": format_name,
        "round": "operational",
        "candidate": candidate,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(),
            "rust_toolchain": RUST_TOOLCHAIN or "active",
        },
        "build_seconds": build_seconds,
        "production_build_seconds": production_build_seconds,
        "dependency_count": dependency_count,
        "binary_size_bytes": runner_binary_size,
        "production_dependency_count": production_dependency_count,
        "production_binary_sizes": production_binaries,
        "throughput_bytes_per_second": (
            total_input_bytes / (total_elapsed_ns / 1_000_000_000) if total_elapsed_ns else None
        ),
        "timeout_seconds": TIMEOUT_SECONDS,
        "cases": cases,
        "gate_pass": all(case["gate_pass"] for case in cases if case["selection_scope"] or not case["valid"]),
        "rss_note": "Measured with /usr/bin/time on POSIX; unavailable on Windows.",
    }
    output_dir = ROOT / ".ai" / "process" / "extractor-evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {format_name.upper()} extractor operational report",
        "",
        f"- Run: `{run_id}`",
        f"- Candidate: `{candidate}`",
        f"- Platform: `{report['environment']['platform']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Build: `{build_seconds:.2f}s`",
        f"- Production workspace build: `{production_build_seconds:.2f}s`",
        f"- Runner dependencies: `{dependency_count}` packages",
        f"- Production dependencies: `{production_dependency_count}` packages",
        f"- Runner release binary: `{runner_binary_size} bytes`",
        f"- Production binaries: `{production_binaries}`",
        "- Production binary size delta: `not available (no pre-dependency baseline)`",
        f"- Throughput: `{report['throughput_bytes_per_second'] or 0:.0f} bytes/s`",
        "",
        "| Fixture | Expected | Statuses | p50 (ms) | p95 (ms) | Peak RSS (KiB) | Truth | Gate |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in cases:
        expected = "ok" if case["valid"] else "error"
        statuses = ", ".join(case["statuses"])
        p50 = "—" if case["p50_ms"] is None else f"{case['p50_ms']:.2f}"
        p95 = "—" if case["p95_ms"] is None else f"{case['p95_ms']:.2f}"
        rss = "—" if case["rss_peak_kib"] is None else str(case["rss_peak_kib"])
        lines.append(
            f"| {case['fixture_id']} | {expected} | {statuses} | {p50} | {p95} | {rss} | "
            f"{'PASS' if case['truth_pass'] else 'FAIL'} | "
            f"{'PASS' if case['gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"## Gate: `{'PASS' if report['gate_pass'] else 'FAIL'}`",
            "",
            "The gate requires every in-scope valid fixture to parse, every invalid fixture to return an error, "
            "no crash/protocol error/timeout, and no fixed performance or size threshold.",
            "",
            report["rss_note"],
            "",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print("\n".join(lines))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected extractor operational measurements")
    parser.add_argument("--format", choices=("docx", "pdf"), required=True)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    run_operational(args.format, iterations=args.iterations)


if __name__ == "__main__":
    main()
