"""The ``survey`` command: detect workspace changes and report stale files.

Detection only — never touches the embedder or creates a database. With no
index yet, it classifies the workspace against an empty snapshot (every
supported file is ``new``), so a user can see what ``mine`` would index
before running it.
"""

from __future__ import annotations

import typer

import lode.cli as _cli
from lode.cli.commands._common import (
    ConfigArg,
    NoColorArg,
    PaletteArg,
    load_settings_or_fail,
    open_store,
    render_options,
    workspace_from,
)
from lode.cli.render.output import echo_json, json_ok
from lode.ingestion.pipeline import classify, detect_changes


def register(app: typer.Typer) -> None:
    app.command("survey")(survey)
    app.command("status", hidden=True)(survey)


def survey(
    ctx: typer.Context,
    config: ConfigArg = None,
    palette: PaletteArg = None,
    no_color: NoColorArg = False,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Detect workspace changes and report stale files.

    This command only compares files and does not require an embedding endpoint.
    """
    workspace = workspace_from(ctx)
    settings = load_settings_or_fail(config, command="survey", as_json=as_json)
    options = render_options(settings, palette, no_color)
    store = open_store(workspace, None, command="survey", as_json=as_json, tokenizer=settings.fts.strategy)
    if store is None:
        # No index yet: classify against an empty snapshot (all new),
        # without creating a database or touching the embedder.
        result = classify({}, workspace, settings.app.ignore.sources)
    else:
        with store:
            result = detect_changes(store, workspace, settings.app.ignore.sources)
    if as_json:
        echo_json(
            json_ok(
                "survey",
                workspace=workspace.as_posix(),
                summary={
                    "unchanged": result.unchanged,
                    "new": result.new,
                    "changed": result.changed,
                    "missing": result.missing,
                    "renamed": result.renamed,
                    "skipped": result.skipped,
                    "pending": result.pending,
                },
                paths={
                    "new": [str(path) for path in result.new_files],
                    "changed": [str(path) for path in result.changed_files],
                    "missing": [str(path) for path in result.missing_files],
                    "renamed": [{"from": str(old), "to": str(new)} for old, new in result.renamed_files],
                    "unchanged": [str(path) for path in result.unchanged_files],
                },
            )
        )
    else:
        _cli.render_survey(workspace, result, options=options)
