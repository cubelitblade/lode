"""Manifest-backed public and private corpora for extractor evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
DOCX_MANIFEST = ROOT / "evals" / "document_extractors" / "corpora" / "docx-apache-poi.json"
PDF_MANIFESTS = (
    ROOT / "evals" / "document_extractors" / "corpora" / "pdf-pdfjs.json",
    ROOT / "evals" / "document_extractors" / "corpora" / "pdf-verapdf.json",
)


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One corpus document with a report-safe identifier."""

    sample_id: str
    path: Path
    source: str


@dataclass(frozen=True, slots=True)
class CorpusMetadata:
    """Public corpus provenance copied into the evaluation report."""

    name: str
    revision: str
    license: str
    license_url: str
    source_url: str


class CorpusFileEntry(TypedDict):
    name: str
    sha256: str
    url_path: NotRequired[str]


class CorpusManifest(TypedDict):
    corpus: str
    revision: str
    license: str
    license_url: str
    source_url: str
    base_url: str
    cache_dir: NotRequired[str]
    files: list[CorpusFileEntry]


def load_docx_public_corpus(directory: Path | None = None) -> tuple[CorpusMetadata, list[CorpusDocument]]:
    """Download or verify the pinned Apache POI DOCX quality corpus."""
    payload = _read_manifest(DOCX_MANIFEST)
    target = directory or ROOT / ".ai" / "cache" / "document-extractors" / "apache-poi-docx"
    target.mkdir(parents=True, exist_ok=True)
    documents: list[CorpusDocument] = []
    for index, item in enumerate(payload["files"], start=1):
        name = str(item["name"])
        expected_sha256 = str(item["sha256"])
        path = target / name
        if not path.exists() or _sha256(path) != expected_sha256:
            if directory is not None:
                raise ValueError(f"corpus file missing or checksum mismatch: {name}")
            _download(f"{payload['base_url']}/{name}", path)
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual_sha256}")
        documents.append(CorpusDocument(sample_id=f"public-{index:03d}", path=path, source="public"))
    metadata = CorpusMetadata(
        name=str(payload["corpus"]),
        revision=str(payload["revision"]),
        license=str(payload["license"]),
        license_url=str(payload["license_url"]),
        source_url=str(payload["source_url"]),
    )
    return metadata, documents


def load_private_docx_corpus(directory: Path | None = None) -> list[CorpusDocument]:
    """Find local private DOCX files without exposing their names in reports."""
    configured = directory or Path(
        os.environ.get(
            "LODE_EVAL_PRIVATE_DOCX_DIR",
            ROOT / ".ai" / "evals" / "document-extractors" / "private" / "docx",
        )
    )
    if not configured.exists():
        return []
    paths = sorted(path for path in configured.rglob("*") if path.is_file() and path.suffix.lower() == ".docx")
    return [
        CorpusDocument(sample_id=f"private-{index:03d}", path=path, source="private")
        for index, path in enumerate(paths, start=1)
    ]


def load_pdf_public_corpus(
    directory: Path | None = None,
) -> tuple[list[CorpusMetadata], list[CorpusDocument]]:
    """Download or verify pinned PDF.js and veraPDF quality subsets."""
    metadata: list[CorpusMetadata] = []
    documents: list[CorpusDocument] = []
    for manifest_path in PDF_MANIFESTS:
        payload = _read_manifest(manifest_path)
        cache_dir = str(payload.get("cache_dir", manifest_path.stem))
        target_root = directory or ROOT / ".ai" / "cache" / "document-extractors" / "pdf"
        target = target_root / cache_dir
        target.mkdir(parents=True, exist_ok=True)
        corpus_slug = cache_dir.replace("_", "-")
        for index, item in enumerate(payload["files"], start=1):
            name = str(item["name"])
            url_path = str(item.get("url_path", name))
            expected_sha256 = str(item["sha256"])
            path = target / name
            if not path.exists() or _sha256(path) != expected_sha256:
                if directory is not None:
                    raise ValueError(f"corpus file missing or checksum mismatch: {cache_dir}/{name}")
                encoded_path = quote(url_path, safe="/")
                _download(f"{payload['base_url']}/{encoded_path}", path)
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for {cache_dir}/{name}: {actual_sha256}")
            documents.append(
                CorpusDocument(
                    sample_id=f"public-{corpus_slug}-{index:03d}",
                    path=path,
                    source="public",
                )
            )
        metadata.append(
            CorpusMetadata(
                name=str(payload["corpus"]),
                revision=str(payload["revision"]),
                license=str(payload["license"]),
                license_url=str(payload["license_url"]),
                source_url=str(payload["source_url"]),
            )
        )
    return metadata, documents


def load_private_pdf_corpus(directory: Path | None = None) -> list[CorpusDocument]:
    """Find local private PDF files without exposing their names in reports."""
    configured = directory or Path(
        os.environ.get(
            "LODE_EVAL_PRIVATE_PDF_DIR",
            ROOT / ".ai" / "evals" / "document-extractors" / "private" / "pdf",
        )
    )
    if not configured.exists():
        return []
    paths = sorted(path for path in configured.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    return [
        CorpusDocument(sample_id=f"private-{index:03d}", path=path, source="private")
        for index, path in enumerate(paths, start=1)
    ]


def _read_manifest(path: Path) -> CorpusManifest:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid corpus manifest: {path}")
    mapping = cast(dict[str, object], payload)
    if not isinstance(mapping.get("files"), list):
        raise ValueError(f"invalid corpus manifest: {path}")
    return cast(CorpusManifest, mapping)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
