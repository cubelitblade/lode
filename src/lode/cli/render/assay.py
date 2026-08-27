"""Human-readable `lode assay` rendering.

One module per command so the render layer does not grow unbounded. Every
render function takes a ``RenderOptions``; defaults reproduce the current rich
output so callers that do not care about styling get the existing behaviour.

``assay why`` explains why one chunk scored as it did in four sections:
Result (final rank/score plus ranking factors), Evidence (per-source facts),
Fusion (the formula with the actual numbers substituted), and Pipeline (the
static configuration). Everything is data-driven from ``ScoreExplanation``
and the plan's ``params``, so swapping operators needs no render changes.

``assay how`` explains how a chunk's text is processed by the configured
tokenizer: provenance, tokenizer metadata, a text preview, and the actual
token stream as the index side stores it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from rich.cells import cell_len
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from lode.cli.render.core import Intent, RenderOptions
from lode.index import ChunkWithPath
from lode.index.explanation import RetrievalStatus, ScoreExplanation
from lode.lexical.base import IndexedTerm, distinct_terms

# Human-output caps so a large chunk cannot flood the terminal; ``--json``
# always carries the full text and term stream. The term cap is in display
# cells (visual units), not term count.
HOW_TERMS_MAX_CELLS = 200
HOW_TEXT_PREVIEW_CHARS = 600

# Section separator for the four-part `why` report.
_SECTION_RULE = "─" * 64

_SOURCE_TITLES = {"semantic": "Semantic similarity", "lexical": "Lexical relevance"}
_SOURCE_METHODS = {"semantic": "cosine", "lexical": "BM25"}


def render_assay(
    explanation: ScoreExplanation,
    *,
    query: str,
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render a human-readable score explanation in four sections.

    Result → Evidence → Fusion → Pipeline, separated by rules and wrapped in
    a bordered panel (matching the other commands). ``console`` may be
    injected (e.g. a recording console) to capture output in tests.
    """
    if options is None:
        options = RenderOptions()
    console = console or Console(no_color=options.no_color)

    body = Group(
        *_result_section(explanation, query, options),
        _SECTION_RULE,
        "",
        *_evidence_section(explanation, options),
        _SECTION_RULE,
        "",
        *_fusion_section(explanation, options),
        _SECTION_RULE,
        "",
        *_pipeline_section(explanation, options),
    )

    short_id = explanation.chunk.digest.removeprefix("blake3:")[:12]
    frame = options.box
    if frame is not None:
        console.print(
            Panel(
                body,
                title=f"Assay · {short_id}",
                title_align="left",
                border_style=options.border_style,
                box=frame,
            )
        )
    else:
        for part in body.renderables:
            if isinstance(part, Text):
                console.print(part, markup=False, highlight=False, soft_wrap=True)
            elif isinstance(part, Group):
                for sub in part.renderables:
                    console.print(sub, markup=False, highlight=False, soft_wrap=True)
            else:
                console.print(part)


def _result_section(
    explanation: ScoreExplanation,
    query: str,
    options: RenderOptions,
) -> list[Text | Group | str]:
    """Query line, final rank/score, chunk provenance, and ranking factors."""
    info_style = options.intent_colors.get(Intent.INFO, "")
    success_style = options.intent_colors.get(Intent.SUCCESS, "")
    chunk = explanation.chunk

    rank_text = f"{explanation.rank} / {explanation.top_k}" if explanation.rank is not None else "not ranked"
    parts: list[Text | Group | str] = [
        Text(f"  Query: {query}", style=info_style),
        "",
        Text("  Result:", style=info_style),
        Text(f"    Rank:  {rank_text}", style=success_style),
        Text(f"    Score: {explanation.combined:.4f}", style=success_style),
        "",
        Text("  Sources:", style=info_style),
        Text(f"    {_source_label(chunk)}", style=info_style),
        "",
        Text("  Ranking factors:", style=info_style),
        Group(*_ranking_factor_lines(explanation, options)),
    ]
    return parts


def _ranking_factor_lines(explanation: ScoreExplanation, options: RenderOptions) -> list[Text]:
    """One neutral line per participating source, then how they were fused.

    Ranking factor data comes from ``Fusion.explain()`` — no isinstance
    checks on concrete fusion types.
    """
    success_style = options.intent_colors.get(Intent.SUCCESS, "")
    fusion_expl = explanation.plan.fusion.explain(
        explanation.sources,
        explanation.combined,
        explanation.plan.norm,
    )
    ranking_factors = fusion_expl.ranking_factors
    lines: list[Text] = []
    for name in explanation.sources:
        if name in ranking_factors:
            factor = ranking_factors[name]
            # Ranks are whole numbers; contributions carry 4 decimals.
            value_text = f"{factor.value:.0f}" if factor.value.is_integer() else f"{factor.value:.4f}"
            lines.append(Text(f"    {name} {factor.metric} {value_text}", style=success_style))
        else:
            lines.append(Text(f"    {name}: {explanation.sources[name].status.label}"))
    lines.append(Text(f"    fused with {explanation.plan.fusion.name}"))
    return lines


def _evidence_section(explanation: ScoreExplanation, options: RenderOptions) -> list[Text | Group | str]:
    """Per-source facts: method, candidate pool, rank, and score transform.

    Evidence data comes from ``Fusion.explain()`` — no isinstance checks on
    concrete fusion types.
    """
    info_style = options.intent_colors.get(Intent.INFO, "")
    success_style = options.intent_colors.get(Intent.SUCCESS, "")
    fusion_expl = explanation.plan.fusion.explain(
        explanation.sources,
        explanation.combined,
        explanation.plan.norm,
    )
    evidence = fusion_expl.evidence

    entries: list[Text | Group | str] = [Text("  Evidence", style=info_style), ""]
    for name in explanation.sources:
        source = explanation.sources[name]
        entries.append(Text(f"    \u25aa {_SOURCE_TITLES[name]}", style=info_style))
        if source.status is not RetrievalStatus.MATCHED:
            entries.append(Text(f"        status: {source.status.label}"))
            continue
        entries.extend(
            [
                Text(f"        method: {_SOURCE_METHODS[name]}", style=info_style),
                Text(f"        candidates: {source.pool_size}"),
            ]
        )
        block = evidence.get(name)
        if block is not None:
            # Evidence block present — format based on which fields are set.
            if block.normalization:
                entries.append(Text(f"        rank: {block.rank}"))
                entries.append(
                    Text(
                        f"        score: {block.raw_score:.4f} -> {block.prepared_score:.4f} ({block.normalization})",
                        style=info_style,
                    )
                )
            else:
                entries.append(Text(f"        rank: {block.rank}", style=info_style))
                entries.append(Text(f"        score: {block.raw_score:.4f}"))
            if block.weight is not None and block.contribution is not None:
                entries.extend(
                    [
                        Text(""),
                        Text(f"        factor: {block.weight}"),
                        Text(f"        contribution: {block.contribution:.4f}", style=success_style),
                    ]
                )
        else:
            # No evidence block (RRF) — show rank and raw score directly.
            entries.append(Text(f"        rank: {source.pool_rank}", style=info_style))
            entries.append(Text(f"        score: {source.raw_score:.4f}"))
        entries.append("")
    return entries


def _fusion_section(explanation: ScoreExplanation, options: RenderOptions) -> list[Text | Group | str]:
    """The fusion formula with this query's actual numbers substituted.

    Formula data comes from ``Fusion.explain()`` — no isinstance checks on
    concrete fusion types.
    """
    info_style = options.intent_colors.get(Intent.INFO, "")
    success_style = options.intent_colors.get(Intent.SUCCESS, "")
    fusion_expl = explanation.plan.fusion.explain(
        explanation.sources,
        explanation.combined,
        explanation.plan.norm,
    )
    formula = fusion_expl.formula

    parts: list[Text | Group | str] = [
        Text("  Fusion", style=info_style),
        "",
        Text("    \u25aa Method:", style=info_style),
        Text(f"      {formula.method_label}", style=info_style),
        "",
        Text("    \u25aa Calculation:", style=info_style),
        Text(f"      score = {formula.symbolic_terms}", style=info_style),
        Text(f"            = {formula.value_terms}", style=info_style),
        Text(f"            = {formula.result:.4f}", style=success_style),
    ]
    if formula.missing_note:
        parts.append(Text(f"      ({formula.missing_note})"))
    return parts


def _pipeline_section(explanation: ScoreExplanation, options: RenderOptions) -> list[Text | Group | str]:
    """The static retrieval configuration behind the run."""
    info_style = options.intent_colors.get(Intent.INFO, "")
    plan = explanation.plan

    norm_line = (
        f"{plan.norm.name} ({_format_params(plan.norm.params)})" if plan.norm is not None else "none (rank based)"
    )
    fusion_param_lines: list[str] = []
    params = plan.fusion.params
    if "k" in params:
        fusion_param_lines.append(f"k = {params['k']}")
    if "weights" in params:
        weights = cast("Mapping[str, float]", params["weights"])
        for name, weight in weights.items():
            fusion_param_lines.append(f"{name} = {weight}")

    parts: list[Text | Group | str] = [
        Text("  Pipeline", style=info_style),
        "",
        Text("    \u25aa Retrieval:", style=info_style),
        Text("      semantic cosine"),
        Text("      lexical BM25"),
        "",
        Text("    \u25aa Normalization:", style=info_style),
        Text(f"      {norm_line}"),
        "",
        Text("    \u25aa Fusion:", style=info_style),
        Text(f"      {plan.fusion.name} ({', '.join(fusion_param_lines)})"),
    ]
    return parts


def _format_params(params: Mapping[str, object]) -> str:
    """Format operator params as ``key = value`` pairs joined by commas."""
    return ", ".join(f"{key} = {value}" for key, value in sorted(params.items()))


def render_how(
    chunk: ChunkWithPath,
    *,
    tokenizer: str,
    tokenize_clause: str,
    terms: Sequence[IndexedTerm],
    options: RenderOptions | None = None,
    console: Console | None = None,
) -> None:
    """Render how a chunk's text is tokenized at index time.

    Shows the chunk provenance, the active tokenizer metadata, a preview of
    the processed text, and the distinct index terms in their original order.
    The term line is truncated by rendered width (visual units), not by term
    count, so it never floods the terminal regardless of term length;
    ``--json`` always carries the complete stream. ``console`` may be injected
    to capture output in tests.
    """
    if options is None:
        options = RenderOptions()
    console = console or Console(no_color=options.no_color)

    info_style = options.intent_colors.get(Intent.INFO, "")
    muted_style = options.intent_colors.get(Intent.MUTED, "")
    short_id = chunk.digest.removeprefix("blake3:")[:12]

    distinct = distinct_terms(terms)
    term_line = _terms_line(distinct)

    text_preview = chunk.text
    if len(text_preview) > HOW_TEXT_PREVIEW_CHARS:
        text_preview = text_preview[: HOW_TEXT_PREVIEW_CHARS - 3] + "..."

    def _line(label: str, body_text: str, *, style: str) -> Group:
        return Group(
            Text(label, style=info_style),
            Text(body_text, style=style),
        )

    body_parts: list[Text | Group | str] = [
        Text(_source_label(chunk), style=info_style),
        "",
        _line(
            f"Lexical analyzer: {tokenizer}",
            f"Storage tokenizer: {tokenize_clause}",
            style=muted_style,
        ),
        "",
        _line(f"Text ({len(chunk.text)} chars):", text_preview, style=""),
        "",
        _line(f"Terms ({len(distinct)}):", term_line, style=muted_style),
    ]

    frame = options.box
    if frame is not None:
        console.print(
            Panel(
                Group(*body_parts),
                title=f"How · {short_id}",
                title_align="left",
                border_style=options.border_style,
                box=frame,
            )
        )
    else:
        first = True
        for part in body_parts:
            if not first:
                console.print()
            if isinstance(part, Text):
                console.print(part, markup=False, highlight=False, soft_wrap=True)
            elif isinstance(part, Group):
                for sub in part.renderables:
                    console.print(sub, markup=False, highlight=False, soft_wrap=True)
            first = False


def _terms_line(terms: Sequence[IndexedTerm]) -> str:
    """One inline line of terms, truncated by rendered width.

    Terms keep their original order (no frequency sort), joined with `` · ``
    so adjacent CJK surfaces stay visually separate. Truncation counts
    display cells (CJK characters are double-width via rich's cell measure),
    not term count — a few long trigrams truncate just like many short ones.
    """
    if not terms:
        return "(none)"
    separator = " · "

    def _piece(term: IndexedTerm) -> str:
        if not term.variants:
            return term.surface
        return f"{term.surface}({'/'.join(term.variants)})"

    pieces = [_piece(term) for term in terms]
    budget = HOW_TERMS_MAX_CELLS - len(" … more")
    line = ""
    used = 0
    for index, piece in enumerate(pieces):
        part = piece if index == 0 else separator + piece
        width = cell_len(part)
        if used + width > budget and index > 0:
            remaining = len(pieces) - index
            return f"{line} … (+{remaining} more)"
        line += part
        used += width
    return line


def _source_label(chunk: ChunkWithPath) -> str:
    """One-line provenance for a chunk: path(s), heading chain, page."""
    source = chunk.primary.path
    if len(chunk.refs) > 1:
        source += f" (+{len(chunk.refs) - 1} more)"
    if chunk.heading:
        source += f" > {chunk.heading}"
    if chunk.page is not None:
        source += f" (p.{chunk.page})"
    return source
