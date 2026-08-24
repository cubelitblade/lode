"""End-to-end CLI tests: mine -> prospect round trip.

The real embedder (network) is replaced with a FakeEmbedder via
monkeypatch; everything else runs through the actual typer app.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lode import config
from lode.cli import app
from lode.cli.render import PLAIN_INTENT_COLORS, RenderOptions
from lode.config import EmbeddingConfig
from lode.ingestion import chunk_digest
from lode.ingestion.pipeline import DetectResult
from tests.fakes import FailingEmbedder, FakeEmbedder

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


def _fake_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return FakeEmbedder()


def _other_model_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return FakeEmbedder(model_id="other-model")


def _dimension_mismatch_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
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


def _wrong_query_embedder(_cfg: EmbeddingConfig) -> FakeEmbedder:
    return _WrongQueryEmbedder(model_id="test-model", dimension=4)


def _failing_embedder(_cfg: EmbeddingConfig) -> FailingEmbedder:
    return FailingEmbedder()


def test_mine_then_prospect_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(
        "The experiment showed strong quantum entanglement in the third group."
    )

    mine = runner.invoke(app, ["mine", str(tmp_path)])
    assert mine.exit_code == 0, mine.output
    assert "+ added 1" in mine.output

    prospect = runner.invoke(app, ["prospect", "entanglement", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output
    assert "docs/report.txt" in prospect.output
    assert "quantum entanglement" in prospect.output
    # The short chunk id (blake3: prefix stripped) is appended to the line.
    assert "#" in prospect.output
    assert "blake3:" not in prospect.output


def test_survey_reports_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    report = tmp_path / "a.txt"
    report.write_text("hello world")

    runner.invoke(app, ["mine", str(tmp_path)])
    report.write_text("changed content!")

    survey = runner.invoke(app, ["survey", str(tmp_path)])
    assert survey.exit_code == 0, survey.output
    # Human-readable output is presentation, not a stable contract: assert only
    # that the changed file is reported, not the exact formatting.
    assert "a.txt" in survey.output


def test_survey_without_index_reports_all_new_and_creates_no_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Survey with no index classifies everything as new without creating a db.

    It must not touch the embedder (no endpoint needed) and must not create
    the index database — that is mine's job.
    """
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("second file")

    result = runner.invoke(app, ["survey", "--json", str(tmp_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["summary"]["new"] == 2
    assert payload["summary"]["pending"] == 2
    assert payload["summary"]["changed"] == 0
    # No index database was created by a read-only survey.
    assert not (tmp_path / ".lode" / "index.db").exists()


def test_survey_uses_configured_palette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured output.palette flows into the render options."""
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[output]\npalette = "plain"\n')
    runner.invoke(app, ["mine", str(tmp_path)])

    captured: RenderOptions | None = None

    def fake_render(
        workspace: Path,
        result: DetectResult,
        *,
        options: RenderOptions | None = None,
        console: Console | None = None,
    ) -> None:
        nonlocal captured
        captured = options

    monkeypatch.setattr("lode.cli.render_survey", fake_render)
    result = runner.invoke(app, ["survey", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured.intent_colors == PLAIN_INTENT_COLORS


def test_config_show_uses_configured_palette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config output also honours the configured output.palette."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[output]\npalette = "plain"\n')

    captured: RenderOptions | None = None

    def fake_render(
        content: str,
        *,
        options: RenderOptions | None = None,
        console: Console | None = None,
    ) -> None:
        nonlocal captured
        captured = options

    monkeypatch.setattr("lode.cli.render_config_show", fake_render)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured.intent_colors == PLAIN_INTENT_COLORS


def test_mine_from_scratch_flag_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    mine = runner.invoke(app, ["mine", "--from-scratch", str(tmp_path)])
    assert mine.exit_code == 0, mine.output
    assert "+ added 1" in mine.output


def test_mine_with_no_indexable_files_creates_no_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mine with only unsupported files reports Nothing to do. without a db.

    It must not create the index database or touch the embedder when there is
    nothing to embed.
    """
    (tmp_path / "pic.png").write_bytes(b"nope")

    result = runner.invoke(app, ["mine", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Nothing to do." in result.output
    assert not (tmp_path / ".lode" / "index.db").exists()


def test_mine_uses_chunking_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long file is split into more chunks with a small chunk_size."""
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    # Config is loaded relative to the CWD (see lode.toml.example), so run
    # the CLI from the workspace root.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text("[chunking]\nsize = 20\noverlap = 5\n")
    (tmp_path / "a.txt").write_text("word " * 100)

    mine = runner.invoke(app, ["mine", str(tmp_path)])
    assert mine.exit_code == 0, mine.output
    assert "+ added 1" in mine.output

    # With chunk_size=20 the file must have been split into multiple chunks.
    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    assert n_chunks > 1


def test_model_mismatch_blocks_prospect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Same store, different model -> refused until --from-scratch.
    monkeypatch.setattr("lode.cli.build_embedder", _other_model_embedder)
    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code != 0
    assert "different model" in prospect.output

    mine = runner.invoke(app, ["mine", "--from-scratch", str(tmp_path)])
    assert mine.exit_code == 0, mine.output

    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output


def test_search_alias_is_hidden_and_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["search", "hello", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "a.txt" in result.output

    help_out = runner.invoke(app, ["--help"]).output
    # Hidden aliases (status/index/search) must not show in the command list.
    assert "│ search " not in help_out
    assert "│ status " not in help_out
    assert "│ index " not in help_out
    assert "│ survey " in help_out
    assert "│ mine " in help_out
    assert "│ prospect " in help_out


def test_prospect_warns_stale_files_outside_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Change a.txt so survey marks it stale; b.txt stays current.
    (tmp_path / "a.txt").write_text("hello world changed")
    runner.invoke(app, ["survey", str(tmp_path)])

    # top-k 1 keeps only the current hit (b.txt), so the stale file is not
    # in the result set.
    prospect = runner.invoke(app, ["prospect", "entanglement", str(tmp_path), "--top-k", "1"])
    assert prospect.exit_code == 0, prospect.output
    assert "pending changes outside these results" in prospect.output
    assert "Run `lode mine`" in prospect.output


def test_prospect_warns_stale_files_in_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")
    runner.invoke(app, ["mine", str(tmp_path)])

    (tmp_path / "a.txt").write_text("hello world changed")
    runner.invoke(app, ["survey", str(tmp_path)])

    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output
    assert "results include stale files" in prospect.output
    assert "verify them before relying on them" in prospect.output
    assert "Run `lode mine`" in prospect.output


def test_dig_returns_full_chunk_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    digest = chunk_digest(text)
    dig = runner.invoke(app, ["dig", digest, str(tmp_path)])
    assert dig.exit_code == 0, dig.output
    assert "docs/report.txt" in dig.output
    assert text in dig.output
    # Only the short stub is shown, never the full blake3 prefix.
    assert "blake3:" not in dig.output


def test_dig_accepts_short_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    short = chunk_digest(text).removeprefix("blake3:")[:12]
    dig = runner.invoke(app, ["dig", short, str(tmp_path)])
    assert dig.exit_code == 0, dig.output
    assert text in dig.output
    assert "a.txt" in dig.output


def test_dig_accepts_bare_hex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    hex_digest = chunk_digest(text).removeprefix("blake3:")
    dig = runner.invoke(app, ["dig", hex_digest, str(tmp_path)])
    assert dig.exit_code == 0, dig.output
    assert text in dig.output


def test_dig_missing_digest_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    absent = chunk_digest("this text is not indexed")
    dig = runner.invoke(app, ["dig", absent, str(tmp_path)])
    assert dig.exit_code != 0
    assert "Dry hole" in dig.output


def test_dig_without_index_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")  # never mined

    dig = runner.invoke(app, ["dig", "deadbeef", str(tmp_path)])
    assert dig.exit_code != 0
    assert "Dry hole" in dig.output
    assert "run `lode mine`" in dig.output


def test_dig_invalid_digest_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    dig = runner.invoke(app, ["dig", "not-a-digest!", str(tmp_path)])
    assert dig.exit_code != 0
    assert "not a valid digest" in dig.output


def test_dig_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "a.txt").write_text(text)
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["dig", chunk_digest(text), "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "dig"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    window = payload["window"]
    assert window["center_seq"] == 0
    assert window["radius"] == 0
    assert len(window["chunks"]) == 1
    chunk = window["chunks"][0]
    assert chunk["digest"] == chunk_digest(text)
    assert chunk["paths"] == [{"path": "a.txt", "state": "fresh"}]
    assert chunk["heading"] == ""
    assert chunk["page"] is None
    assert chunk["text"] == text


def test_dig_json_accepts_short_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement in the lab"
    (tmp_path / "a.txt").write_text(text)
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    short = chunk_digest(text).removeprefix("blake3:")[:12]
    result = runner.invoke(app, ["dig", short, "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    chunk = payload["window"]["chunks"][0]
    assert chunk["digest"] == chunk_digest(text)
    assert chunk["text"] == text


def test_dig_json_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["dig", chunk_digest("this text is not indexed"), "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "not_found"


def test_dig_json_invalid_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("quantum")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["dig", "not-a-digest!", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "invalid_digest"


def test_dig_json_no_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["dig", "deadbeef", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "no_index"


def _mine_report_with_many_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Mine a single .txt file into several chunks and return one chunk id."""
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    monkeypatch.chdir(tmp_path)
    # Small chunks so one file yields several chunks in a single section.
    (tmp_path / "lode.toml").write_text("[chunking]\nsize = 50\noverlap = 10\n")
    (tmp_path / "report.txt").write_text(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi " * 12
    )
    runner.invoke(app, ["mine", str(tmp_path)])

    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    row = conn.execute("SELECT digest FROM chunks ORDER BY seq LIMIT 1").fetchone()
    conn.close()
    return row[0]


def test_dig_radius_returns_section_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _mine_report_with_many_chunks(tmp_path, monkeypatch)

    result = runner.invoke(app, ["dig", digest, "--radius", "1", str(tmp_path)])
    assert result.exit_code == 0, result.output
    short = digest.removeprefix("blake3:")[:12]
    assert f"Dug {short} with radius 1" in result.output
    # The center chunk is marked and a neighbor card is present.
    assert "0 · center" in result.output
    assert result.output.count("╭─") >= 2


def test_dig_json_radius_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _mine_report_with_many_chunks(tmp_path, monkeypatch)

    result = runner.invoke(app, ["dig", digest, "--radius", "1", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "dig"
    assert payload["success"] is True
    window = payload["window"]
    assert window["center_seq"] == 0
    assert window["radius"] == 1
    assert isinstance(window["chunks"], list)
    assert len(window["chunks"]) >= 1
    center = window["chunks"][0]
    assert center["seq"] == 0
    assert center["digest"].startswith("blake3:")
    assert "text" in center
    assert "heading" in center
    assert "paths" in center
    # The window holds at least one distinct neighbor chunk beyond the center.
    assert any(chunk["digest"] != digest for chunk in window["chunks"])


def test_dig_json_default_single_chunk_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["dig", chunk_digest(text), "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    window = payload["window"]
    assert window["center_seq"] == 0
    assert window["radius"] == 0
    assert len(window["chunks"]) == 1


def test_get_alias_is_hidden_and_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "quantum entanglement"
    (tmp_path / "a.txt").write_text(text)
    runner.invoke(app, ["mine", str(tmp_path)])

    result = runner.invoke(app, ["get", chunk_digest(text), str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert text in result.output

    help_out = runner.invoke(app, ["--help"]).output
    assert "│ get " not in help_out
    assert "│ dig " in help_out


def test_survey_json_reports_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    (tmp_path / "a.txt").write_text("changed content!")
    (tmp_path / "b.txt").write_text("brand new file")

    result = runner.invoke(app, ["survey", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["command"] == "survey"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    assert payload["workspace"] == str(tmp_path)
    assert payload["summary"] == {
        "unchanged": 0,
        "new": 1,
        "changed": 1,
        "missing": 0,
        "skipped": 0,
        "pending": 2,
    }
    assert payload["paths"]["new"] == ["b.txt"]
    assert payload["paths"]["changed"] == ["a.txt"]
    assert payload["paths"]["missing"] == []
    assert payload["paths"]["unchanged"] == []


def test_survey_json_error_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Corrupt the schema version so the store refuses to open.
    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", ("999",))
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["survey", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "survey"
    assert payload["success"] is False
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "schema_version"
    assert "re-mine" in payload["error"]["message"]


def test_mine_json_reports_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("quantum entanglement")

    result = runner.invoke(app, ["mine", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["command"] == "mine"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    assert payload["from_scratch"] is False
    assert payload["workspace"] == str(tmp_path)
    assert payload["summary"] == {"added": 2, "updated": 0, "unchanged": 0, "removed": 0, "skipped": 0}
    assert set(payload["paths"]["added"]) == {"a.txt", "b.txt"}
    assert payload["paths"]["updated"] == []
    assert payload["paths"]["removed"] == []
    assert payload["failed"] == []


def test_mine_from_scratch_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    result = runner.invoke(app, ["mine", "--from-scratch", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["from_scratch"] is True
    assert payload["summary"]["added"] == 1


def test_mine_json_reports_failed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _failing_embedder)
    (tmp_path / "a.txt").write_text("hello world")

    result = runner.invoke(app, ["mine", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["failed"] == [{"path": "a.txt", "error": "embedding endpoint is down"}]
    assert payload["summary"]["added"] == 0


def test_mine_json_model_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    monkeypatch.setattr("lode.cli.build_embedder", _other_model_embedder)
    result = runner.invoke(app, ["mine", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "mine"
    assert payload["success"] is False
    assert payload["error"]["code"] == "model_mismatch"


def test_mine_json_schema_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    conn = sqlite3.connect(str(tmp_path / ".lode" / "index.db"))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", ("999",))
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["mine", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "schema_version"


def test_prospect_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    text = "The experiment showed strong quantum entanglement in the third group."
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text(text)
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["mine", str(tmp_path)])
    result = runner.invoke(app, ["prospect", "entanglement", "--json", "--top-k", "3", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["command"] == "prospect"
    assert payload["success"] is True
    assert payload["schema_version"] == 1
    assert payload["query"] == "entanglement"
    assert payload["top_k"] == 3
    assert len(payload["hits"]) == 1

    hit = payload["hits"][0]
    assert set(hit) == {"rank", "score", "paths", "heading", "page", "digest", "preview"}
    assert hit["rank"] == 1
    assert hit["paths"] == [{"path": "docs/report.txt", "state": "fresh"}]
    assert hit["digest"].startswith("blake3:")
    assert "quantum entanglement" in hit["preview"]
    assert "\n" not in hit["preview"]


def test_prospect_json_preview_flattens_crlf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("line one\r\nline two\rline three\n")
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["mine", str(tmp_path)])
    result = runner.invoke(app, ["prospect", "line", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["hits"]
    preview = payload["hits"][0]["preview"]
    assert "\r" not in preview
    assert "\n" not in preview
    assert preview == "line one line two line three"


def test_prospect_json_empty_hits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")  # never mined -> empty index
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["prospect", "anything", "--json", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["hits"] == []


def test_prospect_json_model_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    monkeypatch.setattr("lode.cli.build_embedder", _other_model_embedder)
    result = runner.invoke(app, ["prospect", "hello", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "prospect"
    assert payload["success"] is False
    assert payload["error"]["code"] == "model_mismatch"


def test_prospect_json_invalid_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["mine", str(tmp_path)])

    # Both retrieval factors zero -> search refuses to run.
    (tmp_path / ".lode" / "config.toml").write_text("[retrieval]\nsemantic_factor = 0.0\nlexical_factor = 0.0\n")

    result = runner.invoke(app, ["prospect", "hello", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["error"]["code"] == "invalid_query"


# -- embedding dimension/model mismatch -----------------------------------------


def test_prospect_dimension_mismatch_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Same model id but a different vector dimension -> gate refuses search.
    monkeypatch.setattr("lode.cli.build_embedder", _dimension_mismatch_embedder)
    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code != 0
    assert "same vein" in prospect.output
    assert "lode mine --from-scratch" in prospect.output


def test_prospect_dimension_mismatch_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    # Reports the stored dimension (4) so the gate passes, but actually emits
    # 99-dim query vectors -> sqlite-vec MATCH throws DimensionMismatchError,
    # which the CLI must translate to the friendly message (Layer 2 fallback).
    monkeypatch.setattr("lode.cli.build_embedder", _wrong_query_embedder)
    result = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert result.exit_code != 0
    assert "same vein" in result.output
    assert "lode mine --from-scratch" in result.output


def test_mine_from_scratch_dimension_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    # A dimension change is allowed with an explicit --from-scratch.
    monkeypatch.setattr("lode.cli.build_embedder", _dimension_mismatch_embedder)
    mine = runner.invoke(app, ["mine", "--from-scratch", str(tmp_path)])
    assert mine.exit_code == 0, mine.output

    # After re-mine the index is 99-dim; prospect with the same embedder works.
    prospect = runner.invoke(app, ["prospect", "hello", str(tmp_path)])
    assert prospect.exit_code == 0, prospect.output


def test_prospect_json_dimension_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lode.cli.build_embedder", _fake_embedder)
    (tmp_path / "a.txt").write_text("hello world")
    runner.invoke(app, ["mine", str(tmp_path)])

    monkeypatch.setattr("lode.cli.build_embedder", _dimension_mismatch_embedder)
    result = runner.invoke(app, ["prospect", "hello", "--json", str(tmp_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["command"] == "prospect"
    assert payload["success"] is False
    assert payload["error"]["code"] == "dimension_mismatch"
    assert "same vein" in payload["error"]["message"]


# -- `lode config` CLI ----------------------------------------------------------


def _config_path(scope: str) -> Path:
    from lode.config import user_config_path, workspace_config_path

    return user_config_path() if scope == "user" else workspace_config_path()


def test_config_show_prints_effective_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[embedding]\nmodel = "m"\n')
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "[embedding]" in result.output
    assert 'model = "m"' in result.output


def test_config_show_subcommand_same_as_bare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    bare = runner.invoke(app, ["config"])
    show = runner.invoke(app, ["config", "show"])
    assert bare.exit_code == 0, bare.output
    assert show.exit_code == 0, show.output
    assert bare.output == show.output


def test_config_set_workspace_creates_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.model", "BAAI/bge-small-zh-v1.5"])
    assert result.exit_code == 0, result.output
    assert ".lode/config.toml" in result.output
    path = _config_path("workspace")
    assert path.is_file()
    assert 'model = "BAAI/bge-small-zh-v1.5"' in path.read_text()


def test_config_set_types_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.batch_size", "8"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["config", "set", "embedding.l2_normalize", "false"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["config", "set", "ignore.sources", ".gitignore, docs"])
    assert result.exit_code == 0, result.output

    data = config.read_toml(_config_path("workspace"))
    assert data["embedding"]["batch_size"] == 8
    assert data["embedding"]["l2_normalize"] is False
    assert data["ignore"]["sources"] == [".gitignore", "docs"]


def test_config_set_writes_existing_project_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lode.toml").write_text('[embedding]\nmodel = "old"\n')
    result = runner.invoke(app, ["config", "set", "embedding.model", "new"])
    assert result.exit_code == 0, result.output
    # Prefers the existing higher-precedence project file (lode.toml), not .lode/config.toml.
    assert "lode.toml" in result.output
    assert '"new"' in (tmp_path / "lode.toml").read_text()


def test_config_set_user_scope_writes_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.api.endpoint", "http://x", "--scope", "user"])
    assert result.exit_code == 0, result.output
    user_path = _config_path("user")
    assert user_path.is_file()
    assert 'endpoint = "http://x"' in user_path.read_text()


def test_config_set_unknown_key_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.unknown", "x"])
    assert result.exit_code != 0
    assert "Dry hole" in result.output
    assert "unknown config key" in result.output


def test_config_set_bad_type_reports_stumbled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "embedding.batch_size", "abc"])
    assert result.exit_code != 0
    assert "Stumbled" in result.output


def test_config_get_reads_merged_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lode").mkdir()
    (tmp_path / ".lode" / "config.toml").write_text('[embedding]\nmodel = "m"\n')
    result = runner.invoke(app, ["config", "get", "embedding.model"])
    assert result.exit_code == 0, result.output
    assert 'embedding.model = "m"' in result.output


def test_config_get_scope_reads_layer_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Only user scope sets the key.
    runner.invoke(app, ["config", "set", "embedding.model", "user-model", "--scope", "user"])
    # Workspace layer has it unset -> dry hole.
    result = runner.invoke(app, ["config", "get", "embedding.model", "--scope", "workspace"])
    assert result.exit_code != 0
    assert "Dry hole" in result.output
    # User layer returns the explicit value.
    result = runner.invoke(app, ["config", "get", "embedding.model", "--scope", "user"])
    assert result.exit_code == 0, result.output
    assert "user-model" in result.output


def test_config_unset_removes_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["config", "set", "embedding.model", "m"])
    result = runner.invoke(app, ["config", "unset", "embedding.model"])
    assert result.exit_code == 0, result.output
    data = config.read_toml(_config_path("workspace"))
    assert "model" not in data.get("embedding", {})
    assert '"model"' not in (tmp_path / ".lode" / "config.toml").read_text()


def test_config_unset_missing_reports_dry_hole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "unset", "embedding.model"])
    assert result.exit_code != 0
    assert "Dry hole" in result.output


def test_config_path_shows_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0, result.output
    assert result.output.strip().endswith(".lode/config.toml")
