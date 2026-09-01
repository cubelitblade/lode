"""Shared fixtures and test doubles for the test suite.

The real embedder (network) is replaced with a FakeEmbedder via monkeypatch;
everything else runs through the actual code under test.
"""

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest
from typer.testing import CliRunner

from lode.config import EmbeddingConfig
from lode.index import Store
from lode.index.ranking import LinearFusion, MinmaxNorm, RetrievalPlan
from lode.ingestion.pipeline import SyncSummary, detect_changes, sync
from lode.ingestion.split import RecursiveSegmentSplitter
from lode.lexical.errors import ExtensionCapability, detect_extension_capability
from tests.fakes import FailingEmbedder, FakeEmbedder, file_record, make_chunks

runner = CliRunner()


def _sqlite_vec_version() -> str:
    """The installed sqlite-vec distribution version, or a fallback marker."""
    try:
        return importlib.metadata.version("sqlite-vec")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _simple_library_path() -> str | Path:
    """The ``simple`` library path for this platform, or a marker."""
    try:
        from lode_simple_native import library_path

        return library_path()
    except ModuleNotFoundError:
        return "not bundled for this platform"


def pytest_report_header(config: pytest.Config, start_path: Path) -> list[str]:  # pyright: ignore[reportUnusedFunction]  # pytest hook: called by the runner, not referenced directly
    """Print the extension-loading capability matrix on every run.

    Lets CI logs show *why* a platform can or cannot load sqlite-vec / the
    ``simple`` tokenizer without needing a dedicated diagnostic step.
    """
    cap = detect_extension_capability()
    return [
        f"python: {cap.python} ({sys.executable})",
        f"sqlite: {cap.sqlite_version}",
        f"enable_load_extension: {cap.can_load}",
        f"SQLITE_OMIT_LOAD_EXTENSION: {'present' if cap.omit_load_extension else 'absent'}",
        f"sqlite-vec: {_sqlite_vec_version()}",
        f"simple library: {_simple_library_path()}",
    ]


@pytest.fixture(autouse=True)
def _isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture: run for every test, not referenced directly
    """Isolate user and project config discovery.

    Redirect the user config dir (``XDG_CONFIG_HOME``) and run from
    ``tmp_path`` so neither the host user config nor the project's own
    ``.lode/config.toml`` / ``lode.toml`` leak into tests (the project config
    now carries e.g. ``app.output.palette``).

    ``XDG_CONFIG_HOME`` only affects platformdirs on posix systems; on Windows
    the user config resolves from ``APPDATA``, so ``user_config_path`` is also
    patched directly to keep tests hermetic on every platform.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    user_path = tmp_path / "user-config" / "config.toml"
    # The CLI binds user_config_path at import time, so patch that reference too.
    monkeypatch.setattr("lode.config.user_config_path", lambda: user_path)
    monkeypatch.setattr("lode.cli.commands.config.user_config_path", lambda: user_path)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def extension_capability() -> ExtensionCapability:
    """Whether this Python's sqlite3 module can load shared extensions.

    Session-scoped so the probe runs once; tests that need native extensions
    (sqlite-vec, the ``simple`` tokenizer) skip when ``can_load`` is False
    instead of failing at fixture setup.
    """
    return detect_extension_capability()


@pytest.fixture
def native_extensions(extension_capability: ExtensionCapability) -> None:
    """Skip a test when the interpreter cannot load native SQLite extensions."""
    if not extension_capability.can_load:
        pytest.skip(extension_capability.skip_reason())


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
    report: Callable[[int, int, PurePosixPath | None], None] | None = None,
) -> SyncSummary:
    """detect then sync — the two-stage shape the CLI now uses."""
    detect = detect_changes(store, tmp_path)
    return sync(store, tmp_path, embedder, SPLITTER, detect=detect, report=report)


def linear_plan(semantic: float, lexical: float) -> RetrievalPlan:
    """A min-max + linear plan with the given weights."""
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
