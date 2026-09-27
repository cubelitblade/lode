//! Human-readable and table output for `assay why`.
//!
//! These renderers consume the core score explanation and its fusion
//! explanation. They do not rerun retrieval or derive scores in the CLI.
#![warn(clippy::pedantic)]

use lode_core::index::explanation::{
    FusionExplanation, RetrievalStatus, ScoreExplanation, SourceExplanation,
};
use lode_core::index::ranking::{Fusion, Norm};
use lode_core::index::records::{ChunkWithRefs, FileStatus, Source};

use crate::render::{Intent, RenderOptions};

const SOURCES: [Source; 2] = [Source::Semantic, Source::Lexical];

/// Render a short explanation for an interactive terminal.
pub(crate) fn compact(explanation: &ScoreExplanation, query: &str, fusion: &FusionExplanation) {
    let options = RenderOptions::stdout();
    let target = &explanation.chunk;
    let primary = target.primary();
    let stale = chunk_stale(target);

    crate::render::print_line(
        options,
        Intent::Info,
        &format!(
            "Assay why · {}",
            crate::assay_render::short_digest(&target.digest)
        ),
    );
    crate::render::print_line(
        options,
        Intent::Info,
        &format!("Query: {}", metadata_controls(query)),
    );
    crate::render::print_line(
        options,
        if stale { Intent::Warning } else { Intent::Info },
        &format!(
            "Target: {}{}",
            metadata_controls(&primary.path.to_native_display()),
            if stale { " [stale]" } else { "" }
        ),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "Score: {} · Rank: {} · top-k: {} · in results: {}",
            format_score(Some(explanation.combined)),
            final_rank(explanation),
            explanation.top_k,
            yes_no(explanation.in_results),
        ),
    );
    crate::render::print_line(options, Intent::Info, "Sources:");
    for source in SOURCES {
        crate::render::print_line(
            options,
            source_intent(explanation.sources[&source].status),
            &compact_source_line(source, &explanation.sources[&source], fusion),
        );
    }
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "Fusion: {} · use `--view extended` for the full explanation.",
            explanation.plan.fusion.clone().name()
        ),
    );
}

/// Render all score and provenance details for an interactive terminal.
pub(crate) fn extended(explanation: &ScoreExplanation, query: &str, fusion: &FusionExplanation) {
    let options = RenderOptions::stdout();
    crate::render::print_line(
        options,
        Intent::Info,
        &format!("Assay why · {}", explanation.chunk.digest),
    );
    crate::render::print_line(
        options,
        Intent::Info,
        &format!("Query: {}", metadata_controls(query)),
    );
    render_target_extended(options, &explanation.chunk);
    render_result_extended(options, explanation);

    crate::render::print_line(options, Intent::Info, "Sources:");
    for source in SOURCES {
        render_source_extended(options, source, &explanation.sources[&source], fusion);
    }
    render_plan_extended(options, explanation);
    render_fusion_extended(options, fusion);
}

fn render_target_extended(options: RenderOptions, target: &ChunkWithRefs) {
    crate::render::print_line(options, Intent::Info, "Target:");
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("  digest: {}", target.digest),
    );
    crate::render::print_line(options, Intent::Muted, "  paths:");
    for reference in &target.refs {
        crate::render::print_line(
            options,
            if reference.status == FileStatus::Stale {
                Intent::Warning
            } else {
                Intent::Muted
            },
            &format!(
                "    {} ({})",
                metadata_controls(&reference.path.to_native_display()),
                reference.status
            ),
        );
    }
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "  heading: {} · page: {} · seq: {}",
            optional_text(&target.heading),
            human_optional_number(target.page),
            human_optional_number(target.seq),
        ),
    );
}

fn render_result_extended(options: RenderOptions, explanation: &ScoreExplanation) {
    crate::render::print_line(options, Intent::Info, "Result:");
    crate::render::print_line(
        options,
        Intent::Success,
        &format!("  combined: {}", format_score(Some(explanation.combined))),
    );
    crate::render::print_line(
        options,
        Intent::Success,
        &format!("  rank: {}", final_rank(explanation)),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "  top-k: {} · in results: {}",
            explanation.top_k,
            yes_no(explanation.in_results)
        ),
    );
}

fn render_plan_extended(options: RenderOptions, explanation: &ScoreExplanation) {
    crate::render::print_line(options, Intent::Info, "Plan:");
    let normalization = if matches!(explanation.plan.fusion, Fusion::Rrf { .. }) {
        "none (skipped by RRF)"
    } else {
        norm_name(explanation.plan.norm)
    };
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "  normalization: {} params={}",
            normalization,
            norm_params(explanation.plan.norm)
        ),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "  fusion: {} params={}",
            explanation.plan.fusion.clone().name(),
            fusion_params(&explanation.plan.fusion)
        ),
    );
}

fn render_fusion_extended(options: RenderOptions, fusion: &FusionExplanation) {
    crate::render::print_line(options, Intent::Info, "Fusion explanation:");
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("  method: {}", fusion.formula.method_label),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "  symbolic: {}",
            metadata_controls(&fusion.formula.symbolic_terms)
        ),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "  values (display rounded): {}",
            metadata_controls(&fusion.formula.value_terms)
        ),
    );
    crate::render::print_line(
        options,
        Intent::Success,
        &format!("  result: {}", format_score(Some(fusion.formula.result))),
    );
    if let Some(note) = &fusion.formula.missing_note {
        crate::render::print_line(options, Intent::Muted, &format!("  note: {note}"));
    }
}

/// Render two stable TSV records, one for each retrieval source.
pub(crate) fn table(explanation: &ScoreExplanation, query: &str, fusion: &FusionExplanation) {
    crate::print_tsv(&[
        "QUERY",
        "TOP_K",
        "DIGEST",
        "PATH",
        "HEADING",
        "PAGE",
        "SEQ",
        "STATE",
        "COMBINED",
        "RANK",
        "IN_RESULTS",
        "SOURCE",
        "STATUS",
        "POOL_SIZE",
        "RAW_SCORE",
        "PREPARED_SCORE",
        "POOL_RANK",
        "FACTOR_METRIC",
        "FACTOR_VALUE",
        "NORM",
        "NORM_PARAMS",
        "FUSION",
        "FUSION_PARAMS",
    ]);

    let target = &explanation.chunk;
    let primary = target.primary();
    let state = if chunk_stale(target) {
        "stale"
    } else {
        "fresh"
    };
    let top_k = explanation.top_k.to_string();
    let page = optional_number(target.page);
    let seq = optional_number(target.seq);
    let combined = explanation.combined.to_string();
    let rank = explanation
        .rank
        .map_or_else(String::new, |value| value.to_string());
    let in_results = explanation.in_results.to_string();
    let norm = norm_name_table(explanation.plan.norm);
    let norm_params = norm_params_table(explanation.plan.norm);
    let fusion_name = explanation.plan.fusion.clone().name().to_string();
    let fusion_params = fusion_params(&explanation.plan.fusion);

    for source in SOURCES {
        let source_explanation = &explanation.sources[&source];
        let factor = fusion.ranking_factors.get(&source);
        let raw_score = optional_float(source_explanation.raw_score);
        let prepared_score = optional_float(source_explanation.prepared_score);
        let pool_rank = source_explanation
            .pool_rank
            .map_or_else(String::new, |value| value.to_string());
        let factor_metric = factor.map_or_else(String::new, |value| value.metric.clone());
        let factor_value = factor.map_or_else(String::new, |value| value.value.to_string());
        let pool_size = source_explanation.pool_size.to_string();
        let fields = [
            query.to_owned(),
            top_k.clone(),
            target.digest.clone(),
            primary.path.as_str().to_owned(),
            target.heading.clone(),
            page.clone(),
            seq.clone(),
            state.to_owned(),
            combined.clone(),
            rank.clone(),
            in_results.clone(),
            source.as_str().to_owned(),
            status_code(source_explanation.status).to_owned(),
            pool_size,
            raw_score,
            prepared_score,
            pool_rank,
            factor_metric,
            factor_value,
            norm.clone(),
            norm_params.clone(),
            fusion_name.clone(),
            fusion_params.clone(),
        ];
        let fields = fields.iter().map(String::as_str).collect::<Vec<_>>();
        crate::print_tsv(&fields);
    }
}

/// Warn human and table users that the explanation uses an old indexed snapshot.
pub(crate) fn stale_warning() {
    let options = RenderOptions::stderr();
    crate::write_stderr(format_args!(
        "{}\n",
        options.paint(
            Intent::Warning,
            "Warning: explaining a stale indexed snapshot; run `lode mine` to update it.",
        )
    ));
}

fn render_source_extended(
    options: RenderOptions,
    source: Source,
    explanation: &SourceExplanation,
    fusion: &FusionExplanation,
) {
    crate::render::print_line(
        options,
        Intent::Info,
        &format!("  {} ({})", source_title(source), source.as_str()),
    );
    crate::render::print_line(
        options,
        source_intent(explanation.status),
        &format!("    status: {}", status_label(explanation.status)),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("    pool size: {}", explanation.pool_size),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!("    raw score: {}", format_score(explanation.raw_score)),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "    prepared score: {}",
            format_score(explanation.prepared_score)
        ),
    );
    crate::render::print_line(
        options,
        Intent::Muted,
        &format!(
            "    pool rank: {}",
            explanation
                .pool_rank
                .map_or_else(|| "—".to_owned(), |value| value.to_string())
        ),
    );
    if let Some(factor) = fusion.ranking_factors.get(&source) {
        crate::render::print_line(
            options,
            Intent::Success,
            &format!(
                "    factor: {} = {}",
                factor.metric,
                format_factor(factor.metric.as_str(), factor.value)
            ),
        );
    }
}

fn compact_source_line(
    source: Source,
    explanation: &SourceExplanation,
    fusion: &FusionExplanation,
) -> String {
    let prefix = source.as_str();
    if let Some(factor) = fusion.ranking_factors.get(&source) {
        return format!(
            "  {prefix}: matched · {}={} · pool rank {} / {}",
            factor.metric,
            format_factor(factor.metric.as_str(), factor.value),
            explanation
                .pool_rank
                .map_or_else(|| "—".to_owned(), |value| value.to_string()),
            explanation.pool_size,
        );
    }
    format!(
        "  {prefix}: {} · pool size {}",
        status_label(explanation.status),
        explanation.pool_size
    )
}

fn source_title(source: Source) -> &'static str {
    match source {
        Source::Semantic => "Semantic similarity",
        Source::Lexical => "Lexical relevance",
    }
}

fn source_intent(status: RetrievalStatus) -> Intent {
    match status {
        RetrievalStatus::Matched => Intent::Success,
        RetrievalStatus::Disabled | RetrievalStatus::Empty | RetrievalStatus::NotRetrieved => {
            Intent::Muted
        }
    }
}

fn status_code(status: RetrievalStatus) -> &'static str {
    match status {
        RetrievalStatus::Disabled => "disabled",
        RetrievalStatus::Empty => "empty",
        RetrievalStatus::NotRetrieved => "not_retrieved",
        RetrievalStatus::Matched => "matched",
    }
}

fn status_label(status: RetrievalStatus) -> &'static str {
    match status {
        RetrievalStatus::Disabled => "disabled",
        RetrievalStatus::Empty => "no results",
        RetrievalStatus::NotRetrieved => "not retrieved",
        RetrievalStatus::Matched => "matched",
    }
}

fn final_rank(explanation: &ScoreExplanation) -> String {
    match explanation.rank {
        Some(rank) if rank <= explanation.top_k as usize => format!("{rank}"),
        Some(rank) => format!("{rank} (outside top-k)"),
        None => "not ranked (no positive score)".to_owned(),
    }
}

fn format_score(value: Option<f64>) -> String {
    let Some(value) = value else {
        return "—".to_owned();
    };
    if value != 0.0 && value.abs() < 0.00005 {
        format!("{value:.4e}")
    } else {
        format!("{value:.4}")
    }
}

fn format_factor(metric: &str, value: f64) -> String {
    if metric == "rank" {
        format!("{value:.0}")
    } else {
        format_score(Some(value))
    }
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_number<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn human_optional_number<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "—".to_owned(), |value| value.to_string())
}

fn optional_text(value: &str) -> String {
    if value.is_empty() {
        "—".to_owned()
    } else {
        metadata_controls(value)
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn chunk_stale(target: &ChunkWithRefs) -> bool {
    target
        .refs
        .iter()
        .any(|reference| reference.status == FileStatus::Stale)
}

fn norm_name(norm: Option<Norm>) -> &'static str {
    norm.map_or("none", Norm::name)
}

fn norm_params(norm: Option<Norm>) -> String {
    crate::assay_norm_json(norm)["params"].to_string()
}

fn norm_name_table(norm: Option<Norm>) -> String {
    norm.map_or_else(String::new, |value| value.name().to_owned())
}

fn norm_params_table(norm: Option<Norm>) -> String {
    norm.map_or_else(String::new, |value| {
        crate::assay_norm_json(Some(value))["params"].to_string()
    })
}

fn fusion_params(fusion: &Fusion) -> String {
    crate::assay_fusion_json(fusion)["params"].to_string()
}

fn metadata_controls(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\n' => r"\n".to_owned(),
            '\t' => r"\t".to_owned(),
            '\r' => r"\r".to_owned(),
            character if character.is_control() => format!(r"\u{{{:x}}}", character as u32),
            character => character.to_string(),
        })
        .collect()
}
