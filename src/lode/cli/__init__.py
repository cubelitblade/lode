"""lode CLI: survey / mine / prospect (aliases: status / index / search).

The mining metaphor carries the narrative (survey = detect, mine = embed,
prospect = search); the aliases keep the interface approachable for
practical use. MCP tools (index_status/reindex/search) are a thin layer on
the same functions — CLI first, MCP later (PLAN M1).
"""

from __future__ import annotations

import typer

from lode.cli.render.dig import render_dig as render_dig  # re-exported for lode.cli.render_dig
from lode.cli.render.mine import render_mine as render_mine  # re-exported for lode.cli.render_mine
from lode.cli.render.prospect import render_prospect as render_prospect  # re-exported for lode.cli.render_prospect
from lode.cli.render.survey import render_survey as render_survey  # re-exported for lode.cli.render_survey
from lode.config import build_embedder as build_embedder  # re-exported for lode.cli.build_embedder

app = typer.Typer(
    name="lode",
    help="lode: turn a workspace of documents into a searchable knowledge lode.",
    no_args_is_help=True,
)


# Register the command modules on the shared app. Imported at the bottom so
# the module-level names above (build_embedder, render_*, ...) are bound
# before the command modules resolve them through `lode.cli`.
from lode.cli.commands import config, dig, mine, prospect, survey  # noqa: E402

survey.register(app)
mine.register(app)
prospect.register(app)
dig.register(app)
app.add_typer(config.config_app, name="config")


if __name__ == "__main__":
    app()
