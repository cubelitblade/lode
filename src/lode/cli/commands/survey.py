"""The ``survey`` command: detect workspace changes and report stale files.

Detection only — never touches the embedder or creates a database. With no
index yet, it classifies the workspace against an empty snapshot (every
supported file is ``new``), so a user can see what ``mine`` would index
before running it.
"""

from __future__ import annotations

from pathlib import Path

import typer

import lode.cli as _cli
from lode.cli.commands._common import (
    ConfigArg,
    SurveyWorkspaceArg,
    open_store,
    render_options,
)
from lode.cli.render.output import echo_json, json_ok
from lode.config import load_settings
from lode.ingestion.pipeline import classify, detect_changes


def register(app: typer.Typer) -> None:
    app.command("survey")(survey)
    app.command("status", hidden=True)(survey)


def survey(
    workspace: SurveyWorkspaceArg = Path("."),
    config: ConfigArg = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Detect workspace changes and report stale files.

    This command only compares files and does not require an embedding endpoint.
    """
    settings = load_settings(config)
    options = render_options(settings)
    store = open_store(workspace, None, command="survey", as_json=as_json)
    if store is None:
        # No index yet: classify against an empty snapshot (all new),
        # without creating a database or touching the embedder.
        result = classify({}, workspace, settings.ignore.sources)
    else:
        with store:
            result = detect_changes(store, workspace, settings.ignore.sources)
    if as_json:
        echo_json(
            json_ok(
                "survey",
                workspace=str(workspace),
                summary={
                    "unchanged": result.unchanged,
                    "new": result.new,
                    "changed": result.changed,
                    "missing": result.missing,
                    "skipped": result.skipped,
                    "pending": result.pending,
                },
                paths={
                    "new": result.new_files,
                    "changed": result.changed_files,
                    "missing": result.missing_files,
                    "unchanged": result.unchanged_files,
                },
            )
        )
    else:
        _cli.render_survey(workspace, result, options=options)