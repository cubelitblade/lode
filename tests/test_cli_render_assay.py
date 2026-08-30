"""Unit tests for `lode.cli.render.assay` (render_assay + render_how).

Human-readable CLI output is presentation, not a stable contract, so these
tests assert behavioural/structural signals rather than exact text or spacing:

* the four sections appear (Result / Evidence / Fusion / Pipeline);
* the Result section shows rank, score, provenance, and ranking factors;
* the Evidence section shows per-source method/pool/rank and the score
  transform, including non-matched statuses;
* the Fusion section substitutes this query's numbers into the formula for
  both linear and RRF shapes;
* the Pipeline section shows retrieval methods, norm params, fusion params;
* `render_how` shows provenance, tokenizer metadata, and the distinct index
  terms with their variants (e.g. pinyin readings).

Colour is deliberately not asserted here — it is stripped in non-TTY test runs
and is an intent/palette concern covered by `lode.cli.render`.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from rich.console import Console

from lode.cli.render import RenderOptions
from lode.cli.render.assay import render_assay, render_how
from lode.index import ChunkWithPath, FileStatus, PathRef
from lode.index.explanation import RetrievalStatus, ScoreExplanation, SourceExplanation
from lode.index.ranking import LinearFusion, MinmaxNorm, RetrievalPlan, RrfFusion, SoftmaxNorm
from lode.lexical.base import IndexedTerm, distinct_terms


def _source(
    *,
    status: RetrievalStatus = RetrievalStatus.MATCHED,
    pool_size: int = 40,
    raw: float | None = 0.9,
    prepared: float | None = 1.0,
    pool_rank: int | None = 1,
) -> SourceExplanation:
    return SourceExplanation(
        status=status,
        pool_size=pool_size,
        raw_score=raw,
        prepared_score=prepared,
        pool_rank=pool_rank,
    )


def _explanation(
    *,
    semantic: SourceExplanation | None = None,
    lexical: SourceExplanation | None = None,
    combined: float = 0.8,
    rank: int | None = 1,
    plan: RetrievalPlan | None = None,
) -> ScoreExplanation:
    if plan is None:
        plan = RetrievalPlan(
            norm=MinmaxNorm(),
            fusion=LinearFusion(weights={"semantic": 0.7, "lexical": 0.3}),
        )
    if semantic is None:
        semantic = _source(pool_size=40)
    if lexical is None:
        lexical = _source(pool_size=10, raw=-5.0, prepared=0.5, pool_rank=3)
    return ScoreExplanation(
        chunk=ChunkWithPath(
            digest="blake3:0123456789abcdef",
            text="quantum entanglement",
            heading="Intro",
            refs=(PathRef(path=PurePosixPath("docs/report.txt"), status=FileStatus.FRESH),),
            page=3,
            seq=2,
        ),
        sources={"semantic": semantic, "lexical": lexical},
        combined=combined,
        rank=rank,
        in_results=True,
        plan=plan,
        top_k=5,
    )


def _render(explanation: ScoreExplanation, options: RenderOptions | None = None) -> str:
    """Render an explanation to plain text via a recording console.

    A wide fixed console keeps line wrapping out of the assertions.
    """
    console = Console(record=True, force_terminal=False, width=200)
    render_assay(explanation, query="entanglement", options=options, console=console)
    return console.export_text()


def test_render_assay_shows_query_and_result() -> None:
    text = _render(_explanation())
    assert "Query: entanglement" in text
    assert "Rank:" in text
    assert "1 / 5" in text
    assert "Score: 0.8000" in text


def test_render_assay_shows_provenance() -> None:
    text = _render(_explanation())
    assert f"{Path('docs') / 'report.txt'} > Intro (p.3)" in text


def test_render_assay_shows_four_sections() -> None:
    text = _render(_explanation())
    assert "Evidence" in text
    assert "Fusion" in text
    assert "Pipeline" in text


def test_render_assay_evidence_shows_source_facts() -> None:
    text = _render(_explanation())
    assert "Semantic similarity" in text
    assert "Lexical relevance" in text
    assert "method: cosine" in text
    assert "method: BM25" in text
    assert "candidates: 40" in text
    assert "candidates: 10" in text
    # The score transform shows raw -> prepared with the norm name.
    assert "0.9000 -> 1.0000 (min-max)" in text
    assert "-5.0000 -> 0.5000 (min-max)" in text


def test_render_assay_linear_fusion_calculation() -> None:
    text = _render(_explanation())
    # Formula with actual numbers substituted.
    assert "semantic × 0.7 + lexical × 0.3" in text  # noqa: RUF001 — intentional multiplication-sign glyph
    assert "1.0000 × 0.7" in text  # noqa: RUF001 — intentional multiplication-sign glyph
    assert "= 0.8000" in text


def test_render_assay_rrf_fusion_calculation() -> None:
    plan = RetrievalPlan(norm=None, fusion=RrfFusion(k=60))
    text = _render(_explanation(plan=plan))
    assert "RRF (k=60)" in text
    assert "1 ÷ (60 + semantic_rank)" in text
    assert "1 ÷ 61" in text


def test_render_assay_rrf_skips_normalization() -> None:
    plan = RetrievalPlan(norm=None, fusion=RrfFusion(k=60))
    text = _render(_explanation(plan=plan))
    assert "none (rank based)" in text
    assert "min-max" not in text


def test_render_assay_ranking_factors_both_shapes() -> None:
    # Linear shape lists contributions; RRF lists ranks.
    linear_text = _render(_explanation())
    assert "Ranking factors:" in linear_text
    assert "semantic contribution 0.7000" in linear_text

    rrf_plan = RetrievalPlan(norm=None, fusion=RrfFusion(k=60))
    rrf_text = _render(_explanation(plan=rrf_plan))
    assert "semantic rank 1" in rrf_text
    assert "fused with rrf" in rrf_text


def test_render_assay_disabled_source() -> None:
    plan = RetrievalPlan(
        norm=MinmaxNorm(),
        fusion=LinearFusion(weights={"semantic": 0.0, "lexical": 1.0}),
    )
    disabled = _source(status=RetrievalStatus.DISABLED, pool_size=0, raw=None, prepared=None, pool_rank=None)
    text = _render(_explanation(semantic=disabled, plan=plan))
    assert "status: disabled" in text


def test_render_assay_not_retrieved_source() -> None:
    not_retrieved = _source(status=RetrievalStatus.NOT_RETRIEVED, pool_size=40, raw=None, prepared=None, pool_rank=None)
    text = _render(_explanation(semantic=not_retrieved))
    assert "status: not retrieved" in text


def test_render_assay_empty_pool_source() -> None:
    empty = _source(status=RetrievalStatus.EMPTY, pool_size=0, raw=None, prepared=None, pool_rank=None)
    text = _render(_explanation(lexical=empty))
    assert "status: no results" in text


def test_render_assay_not_ranked() -> None:
    text = _render(_explanation(combined=0.0, rank=None))
    assert "not ranked" in text


def test_render_assay_pipeline_shows_params() -> None:
    plan = RetrievalPlan(norm=MinmaxNorm(), fusion=LinearFusion(weights={"semantic": 0.7, "lexical": 0.3}))
    text = _render(_explanation(plan=plan))
    assert "semantic cosine" in text
    assert "lexical BM25" in text
    assert "linear" in text
    assert "0.7" in text

    softmax_plan = RetrievalPlan(norm=SoftmaxNorm(temperature=2.0), fusion=RrfFusion(k=30))
    softmax_text = _render(_explanation(plan=softmax_plan))
    assert "softmax (temperature = 2.0)" in softmax_text
    assert "rrf (k = 30)" in softmax_text


def test_render_assay_hides_full_content_address() -> None:
    text = _render(_explanation())
    assert "blake3:" not in text


def test_render_how_shows_provenance_and_tokenizer() -> None:
    chunk = ChunkWithPath(
        digest="blake3:0123456789abcdef",
        text="quantum entanglement",
        heading="Intro",
        refs=(PathRef(path=PurePosixPath("docs/report.txt"), status=FileStatus.FRESH),),
        page=3,
        seq=2,
    )
    console = Console(record=True, force_terminal=False)
    render_how(
        chunk,
        tokenizer="simple",
        tokenize_clause="simple",
        terms=[IndexedTerm("quantum"), IndexedTerm("entanglement")],
        options=RenderOptions(),
        console=console,
    )
    text = console.export_text()
    assert f"{Path('docs') / 'report.txt'} > Intro (p.3)" in text
    assert "Lexical analyzer: simple" in text
    assert "Storage tokenizer: simple" in text
    assert "quantum entanglement" in text
    assert "blake3:" not in text


def test_render_how_shows_variants() -> None:
    chunk = ChunkWithPath(
        digest="blake3:0123456789abcdef",
        text="知识",
        heading="",
        refs=(PathRef(path=PurePosixPath("a.txt"), status=FileStatus.FRESH),),
        seq=0,
    )
    console = Console(record=True, force_terminal=False)
    render_how(
        chunk,
        tokenizer="simple",
        tokenize_clause="simple",
        terms=[IndexedTerm("知", ("z", "zhi")), IndexedTerm("识", ("s", "shi"))],
        options=RenderOptions(),
        console=console,
    )
    text = console.export_text()
    # Variants are folded into the surface instead of polluting the stream.
    assert "知(z/zhi)" in text
    assert "识(s/shi)" in text


def test_render_how_dedupes_repeated_terms() -> None:
    # Dedup semantics live in `distinct_terms`; the render just consumes it.
    merged = distinct_terms([IndexedTerm("dog"), IndexedTerm("chases"), IndexedTerm("dog")])
    assert [term.surface for term in merged] == ["dog", "chases"]


def test_render_how_caps_term_stream_by_width() -> None:
    from lode.cli.render.assay import HOW_TERMS_MAX_CELLS

    total = 400
    chunk = ChunkWithPath(
        digest="blake3:0123456789abcdef",
        text=" ".join(f"w{i}" for i in range(total)),
        heading="",
        refs=(PathRef(path=PurePosixPath("a.txt"), status=FileStatus.FRESH),),
        seq=0,
    )
    console = Console(record=True, force_terminal=False)
    render_how(
        chunk,
        tokenizer="unicode61",
        tokenize_clause="unicode61",
        terms=[IndexedTerm(f"w{i}") for i in range(total)],
        options=RenderOptions(),
        console=console,
    )
    text = console.export_text()
    assert "(+" in text and "more)" in text
    # The last term is not printed: truncation happened before it.
    assert f"w{total - 1}" not in text
    # The line stays within the visual budget (plus the suffix).
    terms_line = next(line for line in text.splitlines() if "more)" in line)
    assert len(terms_line) <= HOW_TERMS_MAX_CELLS + 20


def test_render_how_terms_keep_original_order() -> None:
    chunk = ChunkWithPath(
        digest="blake3:0123456789abcdef",
        text="zeta alpha zeta beta",
        heading="",
        refs=(PathRef(path=PurePosixPath("a.txt"), status=FileStatus.FRESH),),
        seq=0,
    )
    console = Console(record=True, force_terminal=False)
    render_how(
        chunk,
        tokenizer="unicode61",
        tokenize_clause="unicode61",
        terms=[IndexedTerm("zeta"), IndexedTerm("alpha"), IndexedTerm("beta")],
        options=RenderOptions(),
        console=console,
    )
    text = console.export_text()
    # First-seen order, no frequency sort.
    assert "zeta · alpha · beta" in text


def test_render_assay_no_color_keeps_output() -> None:
    text = _render(_explanation(), options=RenderOptions(no_color=True))
    assert "Score: 0.8000" in text
    assert "semantic" in text
