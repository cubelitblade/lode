"""Shared fixtures and test doubles for the test suite.

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual code under test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lode.config import EmbeddingConfig
from lode.index import Store
from lode.index.ranking import LinearFusion, MinmaxNorm, RetrievalPlan
from lode.ingestion.pipeline import SyncSummary, detect_changes, sync
from lode.ingestion.split import RecursiveSegmentSplitter
from tests.fakes import FailingEmbedder, FakeEmbedder, file_record, make_chunks

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture: run for every test, not referenced directly
    """Isolate user and project config discovery.

    Redirect the user config dir (``XDG_CONFIG_HOME``) and run from
    ``tmp_path`` so neither the host user config nor the project's own
    ``.lode/config.toml`` / ``lode.toml`` leak into tests (the project config
    now carries e.g. ``output.palette``).
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    # Keep the db under .lode/ like the real CLI, so the WAL sidecar files
    # are ignored by discover and don't pollute the counts.
    return Store(tmp_path / ".lode" / "index.db", FakeEmbedder())


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


SPLITTER = RecursiveSegmentSplitter(chunk_size=50, chunk_overlap=5)


def run_sync(
    store: Store,
    tmp_path: Path,
    embedder: FakeEmbedder | FailingEmbedder,
    *,
    report: Callable[[int, int, str], None] | None = None,
) -> SyncSummary:
    """detect then sync — the two-stage shape the CLI now uses."""
    detect = detect_changes(store, tmp_path)
    return sync(store, tmp_path, embedder, SPLITTER, detect=detect, report=report)


def linear_plan(semantic: float, lexical: float) -> RetrievalPlan:
    """A min-max + linear plan with the given weights (the default shape)."""
    return RetrievalPlan(
        norm=MinmaxNorm(),
        fusion=LinearFusion(weights={"semantic": semantic, "lexical": lexical}),
    )


@pytest.fixture
def seeded_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "index.db", FakeEmbedder())
    store.replace_file(file_record("a.txt"), *make_chunks(["the quick brown fox jumps", "lazy dog sleeps"]))
    store.replace_file(file_record("b.md", digest="blake3:bb", size=2), *make_chunks(["quantum entanglement in labs"]))
    return store


def fake_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return FakeEmbedder()


def other_model_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return FakeEmbedder(model_id="other-model")


def dimension_mismatch_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    """Same model id but a different reported vector dimension than the index."""
    return FakeEmbedder(model_id="test-model", dimension=99)


class _WrongQueryEmbedder(FakeEmbedder):
    """Reports the stored dimension but emits query vectors of another width.

    Simulates a config dimension that mirrors the stored value while the model
    actually returns a different width — the gate cannot detect it, so the
    query-time fallback must.
    """

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 99


def wrong_query_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return _WrongQueryEmbedder(model_id="test-model", dimension=4)


def failing_embedder(_cfg: EmbeddingConfig) -> FailingEmbedder:
    return FailingEmbedder()
