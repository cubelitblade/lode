use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use terminal_size::{Width, terminal_size};

use lode_core::config::build_embedder;
use lode_core::config::layered::load_settings_for;
use lode_core::index::records::FileRecord;
use lode_core::index::store::Store;
use lode_core::ingestion::pipeline::{
    Change, DetectResult, SyncSummary, classify, detect_changes, sync,
};
use lode_core::ingestion::split::RecursiveSegmentSplitter;
use lode_core::relpath::WorkspacePath;

const HELP_TEMPLATE: &str = "\
Lode {version}

{about}

{usage-heading} {usage}

{all-args}
";

/// Output view for command results (stdout).
///
/// A global abstraction: every command declares which views it supports.
/// `compact` is the default narrative; `extended` adds detail; `table` is
/// the grep-friendly form; `json` is the machine-readable form. Views are
/// orthogonal to `--log-level` (stderr).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum View {
    /// Compact narrative (default).
    Compact,
    /// Detailed narrative with stats and full lists.
    Extended,
    /// Aligned table, optimised for grepping.
    Table,
    /// Machine-readable JSON.
    Json,
}

/// Process log verbosity (stderr).
///
/// Controls how much of the *process* is reported, independent of how the
/// *result* is presented (`--view`). Defaults to `error` so a quiet run
/// stays quiet; `-v`/`-vv` are aliases for `info`/`debug`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

impl LogLevel {
    fn to_level_filter(self) -> log::LevelFilter {
        match self {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
        }
    }
}

/// lode: local-first knowledge mining engine.
#[derive(Parser)]
#[command(
    name = "lode",
    version,
    about = "Local-first knowledge mining engine.",
    long_about = "Turn a workspace of documents into a searchable knowledge lode.",
    help_template = HELP_TEMPLATE
)]
struct Cli {
    /// Workspace to operate on.
    #[arg(short = 'C', long, default_value = ".", global = true)]
    workspace: PathBuf,

    /// Output view for command results (stdout).
    #[arg(long, value_enum, default_value_t = View::Compact, global = true)]
    view: View,

    /// Process log verbosity (stderr).
    #[arg(long, value_enum, global = true)]
    log_level: Option<LogLevel>,

    /// Increase log verbosity (-v = info, -vv = debug).
    #[arg(
        short = 'v',
        action = clap::ArgAction::Count,
        conflicts_with = "log_level",
        global = true
    )]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report ingestion / index status (alias: status).
    #[command(alias = "status")]
    Survey,
    /// Mine / index documents into the store (alias: index).
    Mine {
        /// Discard the existing index (archiving it to `.bak`) and rebuild.
        #[arg(long)]
        from_scratch: bool,
    },
    /// Search the store (alias: search).
    #[command(alias = "search")]
    Prospect {
        /// The query to search for.
        #[arg(value_name = "QUERY")]
        query: String,
        /// Max results to return; defaults to retrieval.top_k from config.
        #[arg(long)]
        top_k: Option<u32>,
    },
    /// Fetch a stored record (alias: get).
    Dig,
    /// Analyze the store: why | how.
    Assay,
    /// Show or edit configuration.
    Config,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    std::process::ExitCode::from(dispatch(cli))
}

/// Route a parsed subcommand to its implementation.
fn dispatch(cli: Cli) -> u8 {
    let log_level = resolve_log_level(cli.verbose, cli.log_level);
    init_logging(log_level);
    match cli.command {
        Command::Survey => survey(&cli.workspace, cli.view),
        Command::Mine { from_scratch } => mine(&cli.workspace, cli.view, from_scratch),
        Command::Prospect { query, top_k } => prospect(&cli.workspace, query, top_k, cli.view),
        Command::Dig => todo_command("dig"),
        Command::Assay => todo_command("assay"),
        Command::Config => todo_command("config"),
    }
}

/// Resolve the effective log level from `-v`/`-vv` and `--log-level`.
///
/// The two are mutually exclusive (`conflicts_with`), so at most one is set:
/// `-v` maps to `info`, `-vv` (or more) to `debug`; otherwise the explicit
/// `--log-level` wins, defaulting to `error`.
fn resolve_log_level(verbose: u8, explicit: Option<LogLevel>) -> LogLevel {
    if verbose > 0 {
        if verbose >= 2 {
            LogLevel::Debug
        } else {
            LogLevel::Info
        }
    } else {
        explicit.unwrap_or(LogLevel::Error)
    }
}

/// Initialise the process logger on stderr.
///
/// Only `lode`'s own modules log at the requested level; third-party crates
/// stay at `error` so dependency internals (e.g. `ignore`'s glob tracing)
/// never flood a debug run.
fn init_logging(level: LogLevel) {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .filter_module("lode", level.to_level_filter())
        .format_timestamp(None)
        .init();
}

/// Not-yet-implemented command: print a TODO on stderr and exit non-zero.
fn todo_command(name: &str) -> u8 {
    eprintln!("`{name}` is not implemented yet");
    1
}

/// Centralised user-facing error text, mirroring the Python `messages.py`
/// convention: every surfaced error pairs a headline with an actionable hint.
///
/// Kept deliberately narrow — only shapes the current commands need. Expand
/// it (rather than scattering ad-hoc `eprintln!`) when a new shape arises.
mod ui_msg {
    /// Write a fatal error to stderr and return a failing exit code.
    ///
    /// Prints `<prefix>: <error>` followed by an indented hint when present.
    pub fn die(command: &str, error: &str, hint: Option<&str>) -> u8 {
        eprintln!("[{command}] {error}");
        if let Some(hint) = hint {
            eprintln!("       Hint: {hint}");
        }
        1
    }

    /// Validate the workspace path: it must exist and be a directory.
    ///
    /// Mirrors the Python app-level callback (`exists=True`,
    /// `file_okay=False`, `dir_okay=True`).
    pub fn bad_workspace(command: &str, ws: &std::path::Path) -> Option<u8> {
        if !ws.exists() {
            return Some(die(
                command,
                &format!("Workspace '{}' does not exist.", ws.display()),
                Some("Point `--workspace/-C` at an existing directory."),
            ));
        }
        if !ws.is_dir() {
            return Some(die(
                command,
                &format!("'{0}' is not a directory.", ws.display()),
                Some("Provide a directory, not a regular file."),
            ));
        }
        None
    }
}

/// The relative path of the index database within the workspace.
const INDEX_DB_RELATIVE: &str = ".lode/index.db";

/// Run the prospect command: search the index with a hybrid query.
///
/// Runs a silent detection first so the stale bits are fresh before search
/// reads them — this command writes `files.status` (it is not read-only).
/// It needs an existing index: with none, it short-circuits with the
/// `no_index` message.
fn prospect(workspace: &std::path::Path, query: String, top_k: Option<u32>, view: View) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("prospect", workspace) {
        return code;
    }
    if query.trim().is_empty() {
        let text = lode_core::messages::require("invalid_query");
        return ui_msg::die("prospect", text.error, text.hint);
    }

    let settings = match load_settings_for(workspace) {
        Ok(s) => s,
        Err(e) => {
            return ui_msg::die(
                "prospect",
                &format!("Could not load configuration: {e}"),
                Some("Check your lode.toml and environment."),
            );
        }
    };
    let embedder = match build_embedder(&settings.embedding) {
        Ok(e) => e,
        Err(e) => {
            return ui_msg::die(
                "prospect",
                &format!("Could not build the embedding client: {e}"),
                Some("Check your embedding configuration in lode.toml."),
            );
        }
    };

    let db_path = workspace.join(INDEX_DB_RELATIVE);
    if !db_path.is_file() {
        let text = lode_core::messages::require("no_index");
        let index_path = db_path.display().to_string();
        let error = lode_core::messages::format(text.error, &[("index_path", index_path)]);
        return ui_msg::die("prospect", &error, text.hint);
    }
    let mut store = match Store::open_existing(&db_path) {
        Ok(s) => s,
        Err(e) => {
            return ui_msg::die(
                "prospect",
                &format!("Cannot open the lode index: {e}"),
                Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
            );
        }
    };

    // Refresh the stale bits before searching so per-chunk annotation and
    // the library-wide dirty signal reflect the current workspace.
    if let Err(e) = detect_changes(&mut store, workspace, &[]) {
        return ui_msg::die(
            "prospect",
            &format!("Could not finish scanning the workspace: {e}"),
            Some("Fix the underlying error and rerun `lode prospect`."),
        );
    }

    let top_k = top_k.unwrap_or(settings.retrieval.top_k);
    if top_k == 0 {
        return ui_msg::die(
            "prospect",
            "--top-k must be at least 1.",
            Some("Set a positive result cap, or drop the flag to use retrieval.top_k."),
        );
    }
    let plan = lode_core::config::build_plan(&settings.retrieval);
    let hits = match lode_core::index::search::search(&store, &*embedder, &query, &plan, top_k) {
        Ok(h) => h,
        Err(e) => {
            let hint = if e.to_string().contains("dimension mismatch") {
                Some("Run `lode mine --from-scratch` to rebuild the index with the current model.")
            } else {
                Some("Fix the underlying error and rerun `lode prospect`.")
            };
            return ui_msg::die(
                "prospect",
                &format!("Could not finish prospecting: {e}"),
                hint,
            );
        }
    };

    match view {
        View::Compact => render_prospect(&hits),
        View::Extended => render_prospect_extended(&hits),
        View::Table => render_prospect_table(&hits),
        View::Json => emit_prospect_json(&query, top_k, &hits),
    }
    0
}

/// One-line preview of a chunk text: whitespace runs collapsed, 160 chars
/// max. Shared by the human views and the JSON payload so the two never
/// drift.
const PREVIEW_MAX_CHARS: usize = 160;

fn preview(text: &str) -> String {
    let mut snippet = String::new();
    let mut last_ws = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            last_ws = true;
            continue;
        }
        if last_ws && !snippet.is_empty() {
            snippet.push(' ');
        }
        last_ws = false;
        snippet.push(ch);
        if snippet.chars().count() >= PREVIEW_MAX_CHARS - 3 {
            break;
        }
    }
    if text.chars().count() > snippet.chars().count() {
        snippet.push_str("...");
    }
    snippet
}

/// The 12-hex short id `prospect` prints and `dig` accepts.
fn short_id(digest: &str) -> &str {
    digest
        .strip_prefix("blake3:")
        .unwrap_or(digest)
        .get(..12)
        .unwrap_or(digest)
}

/// Source line for one hit: primary path, heading chain, page, stale tag,
/// and the extra-reference count.
fn hit_source_line(hit: &lode_core::index::search::SearchHit) -> String {
    let primary = hit.primary();
    let mut line = primary.path.to_native().to_string_lossy().into_owned();
    if hit.refs.len() > 1 {
        line += &format!(" (+{})", hit.refs.len() - 1);
    }
    if !hit.heading.is_empty() {
        line += &format!(" > {}", hit.heading);
    }
    if let Some(page) = hit.page {
        line += &format!(" (p.{page})");
    }
    if primary.status == lode_core::index::records::FileStatus::Stale {
        line += " [stale]";
    }
    line
}

/// Run the survey command: detect workspace changes and report stale files.
///
/// Detection only — never touches the embedder or creates a database. It
/// does flip changed files to stale in an existing index (a write), so the
/// store is opened mutably. With no index yet, it classifies the workspace
/// against an empty snapshot (every supported file is `new`), so a user can
/// see what `mine` would index before running it.
///
/// Supports all four views: `compact` (narrative), `extended` (detailed),
/// `table` (grep-friendly), and `json` (machine-readable).
fn survey(workspace: &std::path::Path, view: View) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("survey", workspace) {
        return code;
    }

    let db_path = workspace.join(INDEX_DB_RELATIVE);
    let has_index = db_path.is_file();
    log::debug!("index database: {}", db_path.display());

    let result = if has_index {
        match Store::open_existing(&db_path) {
            Ok(mut store) => match detect_changes(&mut store, workspace, &[]) {
                Ok(result) => {
                    log::info!("detected {} pending changes", result.pending());
                    result
                }
                Err(e) => {
                    return ui_msg::die(
                        "survey",
                        &format!("Could not finish scanning the workspace: {e}"),
                        Some("Fix the underlying error and rerun `lode survey`."),
                    );
                }
            },
            Err(e) => {
                return ui_msg::die(
                    "survey",
                    &format!("Cannot open the lode index: {e}"),
                    Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
                );
            }
        }
    } else {
        // No index yet: classify against an empty snapshot (all new).
        log::info!("no index found; classifying against an empty snapshot");
        let indexed: HashMap<WorkspacePath, FileRecord> = HashMap::new();
        classify(&indexed, workspace, &[])
    };

    match view {
        View::Compact => render_survey(&result, has_index),
        View::Extended => render_survey_extended(&result, has_index),
        View::Table => render_survey_table(&result, has_index),
        View::Json => emit_json(&result),
    }
    0
}

/// Run the mine command: index new or changed files into the store.
///
/// This is the only command that creates the index database. With no index
/// yet it classifies against an empty snapshot first: if there is nothing
/// to embed it reports "Nothing to do." without creating a database;
/// otherwise it creates the index and syncs.
///
/// `--from-scratch` archives the existing index to `.bak` and rebuilds it
/// under the current configuration, mirroring Python's `reset_index`. The
/// embedder is validated first so a broken endpoint leaves the old index
/// intact.
///
/// 1c scope: extraction + chunking + embedding. The vector dimension comes
/// from the embedder: `embedding.model_dimension` wins when configured,
/// otherwise the embedder probes the endpoint (a single "ping" embed) to
/// infer it — mirroring Python's lazy `dimension` property.
fn mine(workspace: &std::path::Path, view: View, from_scratch: bool) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("mine", workspace) {
        return code;
    }

    let settings = match load_settings_for(workspace) {
        Ok(s) => s,
        Err(e) => {
            return ui_msg::die(
                "mine",
                &format!("Could not load configuration: {e}"),
                Some("Check your lode.toml and environment."),
            );
        }
    };
    let embedder = match build_embedder(&settings.embedding) {
        Ok(e) => e,
        Err(e) => {
            return ui_msg::die(
                "mine",
                &format!("Could not build the embedding client: {e}"),
                Some("Check your embedding configuration in lode.toml."),
            );
        }
    };
    let tokenizer = settings.fts.strategy.clone();
    let splitter =
        match RecursiveSegmentSplitter::new(settings.chunking.size, settings.chunking.overlap) {
            Ok(s) => s,
            Err(e) => {
                return ui_msg::die(
                    "mine",
                    &format!("Invalid chunking configuration: {e}"),
                    Some("Ensure chunking.overlap < chunking.size and chunking.size > 0."),
                );
            }
        };

    let db_path = workspace.join(INDEX_DB_RELATIVE);
    let mut has_index = db_path.is_file();
    log::debug!("index database: {}", db_path.display());

    // `--from-scratch`: archive the existing index and rebuild it under the
    // current configuration. The embedder is validated first (mirroring
    // Python's `reset_index`, which probes `dimension` then `model_id`) so a
    // broken endpoint leaves the old index intact.
    if from_scratch && has_index {
        if let Err(e) = embedder.dimension() {
            return ui_msg::die(
                "mine",
                &format!("Could not determine the embedding dimension: {e}"),
                Some(
                    "Check your embedding endpoint, or set `embedding.model_dimension` in lode.toml.",
                ),
            );
        }
        if let Err(e) = embedder.model_id() {
            return ui_msg::die(
                "mine",
                &format!("Could not determine the embedding model: {e}"),
                Some("Check your embedding endpoint, or set `embedding.model` in lode.toml."),
            );
        }
        match Store::reset(&db_path) {
            Ok(_) => {
                log::info!("archived the previous index to {}.bak", db_path.display());
                // The archive is gone; the create path below rebuilds it.
                has_index = false;
            }
            Err(e) => {
                return ui_msg::die(
                    "mine",
                    &format!("Could not reset the lode index: {e}"),
                    Some("Ensure the workspace is writable."),
                );
            }
        }
    }

    let result = if has_index {
        match Store::open_existing(&db_path) {
            Ok(mut store) => match detect_changes(&mut store, workspace, &[]) {
                Ok(detect) => {
                    let bar = mine_progress_bar(view);
                    let report = mine_report(bar.as_ref());
                    let started = Instant::now();
                    let summary = match sync(
                        &mut store,
                        workspace,
                        &splitter,
                        &detect,
                        Some(&*embedder),
                        report.as_deref(),
                    ) {
                        Ok(summary) => summary,
                        Err(e) => {
                            return ui_msg::die(
                                "mine",
                                &format!("Could not finish mining: {e}"),
                                Some("Fix the underlying error and rerun `lode mine`."),
                            );
                        }
                    };
                    if let Some(bar) = &bar {
                        bar.finish_and_clear();
                    }
                    mine_timed(summary, started)
                }
                Err(e) => {
                    return ui_msg::die(
                        "mine",
                        &format!("Could not finish scanning the workspace: {e}"),
                        Some("Fix the underlying error and rerun `lode mine`."),
                    );
                }
            },
            Err(e) => {
                return ui_msg::die(
                    "mine",
                    &format!("Cannot open the lode index: {e}"),
                    Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
                );
            }
        }
    } else {
        // No index yet: classify against an empty snapshot.
        log::info!("no index found; classifying against an empty snapshot");
        let indexed: HashMap<WorkspacePath, FileRecord> = HashMap::new();
        let detect = classify(&indexed, workspace, &[]);
        if detect.pending() == 0 {
            // Nothing to do — do not create a database.
            SyncSummary {
                unchanged: detect.unchanged.len(),
                skipped: detect.skipped.len(),
                ..Default::default()
            }
        } else {
            // Creating the index needs the embedder's model id and dimension,
            // mirroring Python's `_initialize` (which probes `embedder.dimension`
            // then `embedder.model_id`). Both are only required on the create
            // path — an existing index serves from its stored metadata without
            // touching the endpoint.
            let dimension = match embedder.dimension() {
                Ok(d) => d as u32,
                Err(e) => {
                    return ui_msg::die(
                        "mine",
                        &format!("Could not determine the embedding dimension: {e}"),
                        Some(
                            "Check your embedding endpoint, or set `embedding.model_dimension` in lode.toml.",
                        ),
                    );
                }
            };
            let model_id = match embedder.model_id() {
                Ok(m) => m,
                Err(e) => {
                    return ui_msg::die(
                        "mine",
                        &format!("Could not determine the embedding model: {e}"),
                        Some(
                            "Check your embedding endpoint, or set `embedding.model` in lode.toml.",
                        ),
                    );
                }
            };
            match Store::open(&db_path, &model_id, dimension, &tokenizer) {
                Ok(mut store) => {
                    let bar = mine_progress_bar(view);
                    let report = mine_report(bar.as_ref());
                    let started = Instant::now();
                    let summary = match sync(
                        &mut store,
                        workspace,
                        &splitter,
                        &detect,
                        Some(&*embedder),
                        report.as_deref(),
                    ) {
                        Ok(summary) => summary,
                        Err(e) => {
                            return ui_msg::die(
                                "mine",
                                &format!("Could not finish mining: {e}"),
                                Some("Fix the underlying error and rerun `lode mine`."),
                            );
                        }
                    };
                    if let Some(bar) = &bar {
                        bar.finish_and_clear();
                    }
                    mine_timed(summary, started)
                }
                Err(e) => {
                    return ui_msg::die(
                        "mine",
                        &format!("Cannot create the lode index: {e}"),
                        Some("Ensure the workspace is writable and the configuration is valid."),
                    );
                }
            }
        }
    };

    match view {
        View::Compact => render_mine(workspace, &result),
        View::Extended => render_mine_extended(workspace, &result),
        View::Table => render_mine_table(workspace, &result),
        View::Json => emit_mine_json(&result),
    }
    0
}

/// Build the mine progress bar: an indicatif spinner bar on stderr.
///
/// Suppressed for `--view json` (JSON consumers want a clean stream) and
/// hidden automatically when stderr is not a terminal (indicatif detects
/// this), so `lode mine > log` stays unpolluted.
fn mine_progress_bar(view: View) -> Option<ProgressBar> {
    if view == View::Json {
        return None;
    }
    let style = ProgressStyle::with_template("{spinner:.green} mining {pos}/{len} {msg}")
        .expect("template is static and valid")
        .progress_chars("##-");
    let bar = ProgressBar::new_spinner().with_style(style);
    bar.set_draw_target(ProgressDrawTarget::stderr());
    Some(bar)
}

/// Build the sync progress reporter that drives the bar.
///
/// The bar's length is unknown until sync's first `report` call announces
/// the real total, so the first callback invocation sets it. The message
/// shows the file being processed (OS-native, like every human path).
/// The sync progress callback contract (mirrors `pipeline.rs`'s anonymous
/// signature; factoring it here keeps the helper signatures readable).
///
/// `+ 'a` because the closure borrows the progress bar for as long as the
/// sync pass runs.
type SyncReport<'a> = dyn for<'b> Fn(usize, usize, Option<&'b WorkspacePath>) + 'a;

fn mine_report(bar: Option<&ProgressBar>) -> Option<Box<SyncReport<'_>>> {
    bar.map(|bar| {
        Box::new(
            move |done: usize, total: usize, path: Option<&WorkspacePath>| {
                if bar.length().is_none() && total > 0 {
                    bar.set_length(total as u64);
                }
                bar.set_position(done as u64);
                if let Some(path) = path {
                    bar.set_message(path.to_native().to_string_lossy().into_owned());
                }
            },
        ) as Box<SyncReport<'_>>
    })
}

/// Stamp the wall-clock duration onto the summary.
fn mine_timed(mut summary: SyncSummary, started: Instant) -> SyncSummary {
    summary.duration_seconds = Some(started.elapsed().as_secs_f64());
    summary
}

/// Whether the pass did nothing at all: no processed files and no failures.
fn nothing_done(result: &SyncSummary) -> bool {
    result.added.is_empty()
        && result.updated.is_empty()
        && result.removed.is_empty()
        && result.renamed.is_empty()
        && result.failed.is_empty()
}

/// Number of files that succeeded this run: adds, updates, renames, and
/// removals (i.e. every processed file that did not fail).
fn succeeded_count(result: &SyncSummary) -> usize {
    result.added.len() + result.updated.len() + result.renamed.len() + result.removed.len()
}

/// Processed entries in display order: adds, updates, renames, removals.
fn processed_entries(result: &SyncSummary) -> Vec<(&'static str, String)> {
    let mut rows = Vec::new();
    for path in &result.added {
        rows.push(("added", path.as_str().to_string()));
    }
    for path in &result.updated {
        rows.push(("updated", path.as_str().to_string()));
    }
    for (from, to) in &result.renamed {
        rows.push(("renamed", format!("{} -> {}", from.as_str(), to.as_str())));
    }
    for path in &result.removed {
        rows.push(("removed", path.as_str().to_string()));
    }
    for failure in &result.failed {
        rows.push(("failed", failure.path.as_str().to_string()));
    }
    rows
}

/// Table rows for the mine table: `(STATUS, PATH, DETAIL)`.
/// Table rows for the mine table: `(STATUS, PATH, DETAIL)`.
///
/// A failed row's `DETAIL` is the reason (truncated by the column budget);
/// every other row has nothing more to say than the status itself.
fn mine_table_rows(result: &SyncSummary) -> Vec<(&'static str, String, String)> {
    let mut rows: Vec<(&'static str, String, String)> = Vec::new();
    for path in &result.added {
        rows.push(("added", path.as_str().to_string(), String::new()));
    }
    for path in &result.updated {
        rows.push(("updated", path.as_str().to_string(), String::new()));
    }
    for (from, to) in &result.renamed {
        rows.push((
            "renamed",
            format!("{} -> {}", from.as_str(), to.as_str()),
            String::new(),
        ));
    }
    for path in &result.removed {
        rows.push(("removed", path.as_str().to_string(), String::new()));
    }
    for failure in &result.failed {
        rows.push((
            "failed",
            failure.path.as_str().to_string(),
            failure.error.clone(),
        ));
    }
    rows
}

/// Print the processed list, truncated to `limit` entries with an ellipsis.
fn print_mine_entries(result: &SyncSummary, limit: usize) {
    let entries = processed_entries(result);
    let total = entries.len();
    let shown = total.min(limit);
    for (status, path) in entries.iter().take(shown) {
        let marker = match *status {
            "added" => "+",
            "updated" => "~",
            "renamed" => ">",
            "removed" => "-",
            "failed" => "×",
            _ => " ",
        };
        println!("  {marker} {path}");
    }
    if total > shown {
        println!("  ...");
        println!("  and {} more.", total - shown);
    }
}

/// Print the change list grouped by status, each group with a subtotal.
///
/// Extended view only: the list is untruncated because the user asked for
/// the full picture. Failures are not grouped here — the `Failures` block
/// owns that content.
fn print_grouped(result: &SyncSummary) {
    if !result.added.is_empty() {
        println!("  Added ({}):", result.added.len());
        for path in &result.added {
            println!("    + {}", path.as_str());
        }
    }
    if !result.updated.is_empty() {
        println!("  Updated ({}):", result.updated.len());
        for path in &result.updated {
            println!("    ~ {}", path.as_str());
        }
    }
    if !result.renamed.is_empty() {
        println!("  Renamed ({}):", result.renamed.len());
        for (from, to) in &result.renamed {
            println!("    > {} -> {}", from.as_str(), to.as_str());
        }
    }
    if !result.removed.is_empty() {
        println!("  Removed ({}):", result.removed.len());
        for path in &result.removed {
            println!("    - {}", path.as_str());
        }
    }
}

/// Render a human-readable mine report.
///
/// Mirrors the JSON payload so the numbers never drift. When there is
/// nothing to do, a single "Nothing to do." line is shown instead.
fn render_mine(_workspace: &std::path::Path, result: &SyncSummary) {
    if nothing_done(result) {
        println!("Nothing to do.");
        return;
    }
    let failed = !result.failed.is_empty();
    println!("{}", mine_header(result));
    println!();
    print_mine_entries(result, MAX_LISTED);
    println!();
    let seconds = result.duration_seconds.unwrap_or_default();
    if failed {
        println!(
            "{} succeeded, {} failed.",
            succeeded_count(result),
            result.failed.len()
        );
        println!("Completed in {seconds:.1}s.");
        println!();
        println!("Failures:");
        print_failures(result);
        println!();
        println!("Run `lode mine` again after fixing these issues.");
    } else {
        println!("{} succeeded.", succeeded_count(result));
        println!();
        println!("Completed in {seconds:.1}s.");
        println!();
        println!("The lode is ready for prospecting.");
    }
}

/// The scenario-dependent headline: failures present or not.
fn mine_header(result: &SyncSummary) -> &'static str {
    if result.failed.is_empty() {
        "Mining complete."
    } else {
        "Mining completed with failures."
    }
}

/// Render the detailed (`extended`) mine report.
///
/// Same scenario-dependent headline as `compact`, in the survey extended
/// style: an aligned `Summary` stats block (no-op counts shown only when
/// non-zero), a `Duration` line, and the applied changes grouped by status
/// with inline subtotals, untruncated.
fn render_mine_extended(_workspace: &std::path::Path, result: &SyncSummary) {
    if nothing_done(result) {
        println!("Nothing to do.");
        return;
    }
    let failed = !result.failed.is_empty();
    println!("{}", mine_header(result));
    println!();
    println!("Summary:");
    println!("  {:<8}  {}", "Added:", result.added.len());
    println!("  {:<8}  {}", "Updated:", result.updated.len());
    println!("  {:<8}  {}", "Renamed:", result.renamed.len());
    println!("  {:<8}  {}", "Removed:", result.removed.len());
    if result.unchanged > 0 {
        println!("  {:<8}  {}", "Unchanged:", result.unchanged);
    }
    if failed {
        println!("  {:<8}  {}", "Failed:", result.failed.len());
    }
    println!();
    println!(
        "Duration: {:.1}s",
        result.duration_seconds.unwrap_or_default()
    );

    if succeeded_count(result) > 0 {
        println!();
        println!("Applied changes:");
        println!();
        print_grouped(result);
    }

    if failed {
        println!();
        println!("Failures:");
        print_failures(result);
        println!();
        println!("Run `lode mine` again after fixing these issues.");
    } else {
        println!();
        println!("The lode is ready for prospecting.");
    }
}

/// Print the failure detail block (path + error), shared by compact and
/// extended.
fn print_failures(result: &SyncSummary) {
    for failure in &result.failed {
        println!("  × {}", failure.path.as_str());
        println!("    {}", failure.error);
    }
}

/// Render the mine report as an aligned table, optimised for grepping.
///
/// One row per processed file with `STATUS`/`PATH`/`DETAIL` columns. The
/// table adapts to the terminal width (same budget as the survey table) and
/// empty `DETAIL` renders as `-`.
fn render_mine_table(_workspace: &std::path::Path, result: &SyncSummary) {
    if nothing_done(result) {
        println!("Nothing to do.");
        return;
    }

    println!("{}", mine_header(result));
    println!();

    let rows = mine_table_rows(result);

    // Same column-width budget as the survey table.
    const STATUS_W: usize = 8; // "renamed" is the longest status.
    const COL_GAPS: usize = 4; // two-column gutters.
    const MIN_PATH_W: usize = 28;
    const MAX_PATH_W: usize = 48;
    const MIN_DETAIL_W: usize = 22;
    let term_w = terminal_size()
        .map(|(Width(w), _)| w as usize)
        .unwrap_or(80);

    let widest_path = rows
        .iter()
        .map(|(_, p, _)| p.chars().count())
        .max()
        .unwrap_or(MIN_PATH_W);
    let room_left_over = term_w.saturating_sub(STATUS_W + COL_GAPS + MIN_DETAIL_W);
    let path_w = widest_path
        .clamp(MIN_PATH_W, MAX_PATH_W)
        .min(room_left_over.max(MIN_PATH_W));
    let detail_w = term_w
        .saturating_sub(STATUS_W + COL_GAPS + path_w)
        .min(MIN_DETAIL_W);

    println!(
        "{:<status_w$}  {:<path_w$}  DETAIL",
        "STATUS",
        "PATH",
        status_w = STATUS_W,
        path_w = path_w,
    );
    for (status, path, detail) in &rows {
        let path = truncate_middle(path, path_w);
        let detail = if detail.is_empty() {
            "-".to_string()
        } else {
            truncate_tail(detail, detail_w)
        };
        println!(
            "{:<status_w$}  {:<path_w$}  {}",
            status,
            path,
            detail,
            status_w = STATUS_W,
            path_w = path_w,
        );
    }
}

/// Emit a JSON mine payload.
///
/// Flat and envelope-free, matching the survey JSON convention: just the
/// outcome data, no `ok`/`command`/`workspace` wrapper and no derived counts
/// (a consumer can count the arrays itself). Process metrics
/// (`embedded_chunks`/`duration_seconds`) are presentation data, not
/// outcome data — they stay out of the payload. MCP framing, if any, is
/// assembled by the MCP layer, not the CLI.
fn emit_mine_json(result: &SyncSummary) {
    let payload = serde_json::json!({
        "added": result.added.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        "updated": result.updated.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        "removed": result.removed.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        "renamed": result
            .renamed
            .iter()
            .map(|(f, t)| serde_json::json!({ "from": f.as_str(), "to": t.as_str() }))
            .collect::<Vec<_>>(),
        "failed": result
            .failed
            .iter()
            .map(|f| serde_json::json!({ "path": f.path.as_str(), "error": f.error }))
            .collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

/// Render prospect hits as a flat card stack (compact).
///
/// No header or count line: the hits are visible at a glance and the count
/// is shaped by top_k anyway. Each hit is three lines — rank + source,
/// preview, score · short digest — separated by blank lines. An empty
/// result is a single `Dry hole` line. When the library is dirty, a
/// trailing hint warns where the risk sits.
fn render_prospect(hits: &[lode_core::index::search::SearchHit]) {
    if hits.is_empty() {
        println!("Dry hole: nothing matched.");
        return;
    }
    let preview_w = terminal_preview_width();
    for (index, hit) in hits.iter().enumerate() {
        let rank = index + 1;
        let width = hits.len().to_string().len();
        println!();
        println!("{rank:>width$}  {}", hit_source_line(hit), width = width);
        println!("     {}", truncate_tail(&preview(&hit.text), preview_w));
        println!("     {:.3} · {}", hit.score, short_id(&hit.digest));
    }
    print_prospect_epilogue(hits);
}

/// Usable preview width for the current terminal: the terminal width minus
/// the card indent and a right-side breathing gap, clamped to the 160-char
/// content budget and floored so it never collapses.
fn terminal_preview_width() -> usize {
    const INDENT: usize = 5; // five spaces before the preview line
    const RIGHT_GAP: usize = 2; // breathing room on the right edge
    const MIN_W: usize = 40;
    let term_w = terminal_size()
        .map(|(Width(w), _)| w as usize)
        .unwrap_or(80);
    term_w
        .saturating_sub(INDENT + RIGHT_GAP)
        .clamp(MIN_W, PREVIEW_MAX_CHARS)
}

/// Render prospect hits with full text and every referencing path
/// (extended).
fn render_prospect_extended(hits: &[lode_core::index::search::SearchHit]) {
    if hits.is_empty() {
        println!("Dry hole: nothing matched.");
        return;
    }
    for (index, hit) in hits.iter().enumerate() {
        let rank = index + 1;
        println!("{rank:>3}  {}", hit_source_line(hit));
        if hit.refs.len() > 1 {
            for r in &hit.refs {
                let primary = hit.primary();
                if r.path == primary.path {
                    continue;
                }
                println!("     - {} ({})", r.path.to_native().display(), r.status);
            }
        }
        println!("     {}", hit.text);
        println!("     {:.3} · {}", hit.score, short_id(&hit.digest));
        println!();
    }
    print_prospect_epilogue(hits);
}

/// Render prospect hits as a table (table): RANK/SCORE/DIGEST/PATH.
///
/// heading/page/state stay out: numbers and paths are the grep anchors.
fn render_prospect_table(hits: &[lode_core::index::search::SearchHit]) {
    if hits.is_empty() {
        println!("Dry hole: nothing matched.");
        return;
    }
    const RANK_W: usize = 4;
    const SCORE_W: usize = 6;
    let widest_digest = hits
        .iter()
        .map(|h| short_id(&h.digest).chars().count())
        .max()
        .unwrap_or(12);
    let digest_w = widest_digest.max(6);

    println!(
        "{:<rank_w$}  {:<score_w$}  {:<digest_w$}  PATH",
        "RANK",
        "SCORE",
        "DIGEST",
        rank_w = RANK_W,
        score_w = SCORE_W,
        digest_w = digest_w,
    );
    for (index, hit) in hits.iter().enumerate() {
        let rank = index + 1;
        println!(
            "{:<rank_w$}  {:<score_w$}  {:<digest_w$}  {}",
            rank,
            format!("{:.3}", hit.score),
            short_id(&hit.digest),
            hit_source_line(hit),
            rank_w = RANK_W,
            score_w = SCORE_W,
            digest_w = digest_w,
        );
    }
    print_prospect_epilogue(hits);
}

/// Emit a flat prospect JSON payload (query context + hits + dirty signal).
///
/// Flat and envelope-free like survey/mine; `preview` shares the truncation
/// budget with the human views.
fn emit_prospect_json(query: &str, top_k: u32, hits: &[lode_core::index::search::SearchHit]) {
    let has_stale = hits.iter().any(|h| h.stale());
    let payload = serde_json::json!({
        "query": query,
        "top_k": top_k,
        "has_stale": has_stale,
        "hits": hits.iter().enumerate().map(|(index, hit)| {
            serde_json::json!({
                "rank": index + 1,
                "score": hit.score,
                "paths": hit.refs.iter().map(|r| serde_json::json!({
                    "path": r.path.as_str(),
                    "state": r.status.to_string(),
                })).collect::<Vec<_>>(),
                "heading": hit.heading,
                "page": hit.page,
                "digest": hit.digest,
                "preview": preview(&hit.text),
            })
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

/// Close the prospect narrative: two honest variants depending on where the
/// stale risk sits (a stale hit in this result set, or pending changes
/// elsewhere).
fn print_prospect_epilogue(hits: &[lode_core::index::search::SearchHit]) {
    if !hits.is_empty() && hits.iter().any(|h| h.stale()) {
        println!();
        println!(
            "Warning: results include stale files; verify them before relying on them. Run `lode mine` to update."
        );
    }
}

/// Emit a JSON survey payload.
///
/// The payload is the raw result data — no envelope (`ok`/`command`/
/// `workspace`) and no non-actionable fields (`unchanged`, `skipped`).
/// MCP framing, if any, is assembled by the MCP layer, not the CLI.
fn emit_json(result: &DetectResult) {
    let mut new_paths = Vec::new();
    let mut modified_paths = Vec::new();
    let mut missing_paths = Vec::new();
    let mut renamed_pairs = Vec::new();

    for change in &result.changes {
        match change {
            Change::Added(snap) => new_paths.push(snap.path.as_str()),
            Change::Modified { old, .. } => modified_paths.push(old.path.as_str()),
            Change::Removed(record) => missing_paths.push(record.path.as_str()),
            Change::Renamed { from, to } => renamed_pairs.push((from.as_str(), to.as_str())),
        }
    }

    let payload = serde_json::json!({
        "new": new_paths,
        "modified": modified_paths,
        "missing": missing_paths,
        "renamed": renamed_pairs
            .iter()
            .map(|(f, t)| serde_json::json!({ "from": f, "to": t }))
            .collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

/// Render the survey as an aligned table, optimised for grepping.
///
/// One row per pending change with `STATUS`/`PATH`/`DETAIL` columns. The
/// table adapts to the terminal width: `STATUS` is fixed, `PATH` grows with
/// the longest path (within generous bounds, middles elided), and `DETAIL`
/// gets whatever remains (floored so it never collapses). Empty `DETAIL`
/// renders as `-`.
fn render_survey_table(result: &DetectResult, has_index: bool) {
    let pending = result.pending();
    if pending == 0 {
        if has_index {
            println!("No new findings in this lode.");
        } else {
            println!("The lode is empty — nothing to mine.");
        }
        return;
    }

    // Collect rows: (status, path, detail).
    let mut rows: Vec<(&str, String, String)> = Vec::new();
    for change in &result.changes {
        match change {
            Change::Added(snap) => {
                rows.push(("new", snap.path.as_str().to_string(), String::new()));
            }
            Change::Modified { old, new } => rows.push((
                "modified",
                old.path.as_str().to_string(),
                format!(
                    "{} -> {} bytes, {} -> {}",
                    old.size,
                    new.size,
                    format_unix_ts(old.mtime),
                    format_unix_ts(new.mtime),
                ),
            )),
            Change::Removed(record) => {
                rows.push(("missing", record.path.as_str().to_string(), String::new()));
            }
            Change::Renamed { from, to } => rows.push((
                "renamed",
                format!("{} -> {}", from.as_str(), to.as_str()),
                String::new(),
            )),
        }
    }

    // Column-width budget. STATUS is fixed; PATH scales with the longest
    // path but respects floor/ceiling so neither extreme dominates; DETAIL
    // targets a comfortable width but shrinks on narrow terminals so the
    // table never overflows the available columns.
    const STATUS_W: usize = 8; // "modified" is the longest status.
    const COL_GAPS: usize = 4; // two-column gutters.
    const MIN_PATH_W: usize = 28;
    const MAX_PATH_W: usize = 48;
    const MIN_DETAIL_W: usize = 22;
    let term_w = terminal_size()
        .map(|(Width(w), _)| w as usize)
        .unwrap_or(80);

    let widest_path = rows
        .iter()
        .map(|(_, p, _)| p.chars().count())
        .max()
        .unwrap_or(MIN_PATH_W);
    let room_left_over = term_w.saturating_sub(STATUS_W + COL_GAPS + MIN_DETAIL_W);
    let path_w = widest_path
        .clamp(MIN_PATH_W, MAX_PATH_W)
        .min(room_left_over.max(MIN_PATH_W));
    let detail_w = term_w
        .saturating_sub(STATUS_W + COL_GAPS + path_w)
        .min(MIN_DETAIL_W);

    println!(
        "{:<status_w$}  {:<path_w$}  DETAIL",
        "STATUS",
        "PATH",
        status_w = STATUS_W,
        path_w = path_w,
    );
    for (status, path, detail) in &rows {
        let path = truncate_middle(path, path_w);
        let detail = if detail.is_empty() {
            "-".to_string()
        } else {
            truncate_tail(detail, detail_w)
        };
        println!(
            "{:<status_w$}  {:<path_w$}  {}",
            status,
            path,
            detail,
            status_w = STATUS_W,
            path_w = path_w,
        );
    }
}

/// Elide the middle of a string beyond `max` characters, keeping both the
/// head and the tail: `abc...xyz`.
///
/// Used for the `PATH` column: dropping the interior leaves the leading
/// directories and the trailing filename recognisable, unlike a pure-tail
/// crop which discards the interesting parts wholesale.
///
/// `max` counts Unicode scalar values, not display columns — CJK wide
/// characters count as one. Paths are typically ASCII, so this is a
/// reasonable approximation.
fn truncate_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let avail = max.saturating_sub(3); // reserve space for "..."
    let half = avail / 2;
    let head_end = half;
    let tail_start = chars.len() - (avail - half);
    let head: String = chars[..head_end].iter().collect();
    let tail: String = chars[tail_start..].iter().collect();
    format!("{head}...{tail}")
}

/// Truncate a string to `max` characters, keeping the head.
///
/// Used for the `DETAIL` column: it is auxiliary, so cutting the tail is
/// fine. `max` counts Unicode scalar values, not display columns.
fn truncate_tail(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(3);
    let head: String = chars[..keep].iter().collect();
    format!("{head}...")
}

/// Render a human-readable survey report.
///
/// Three narrative scenarios, each with a clear next step:
/// - no index yet: "No existing lode found." + what `mine` would index
/// - index present, no changes: "No new findings in this lode."
/// - index present, changes: "New findings since last mine:" + pending list
///
/// The list is truncated to [`MAX_LISTED`] entries with an ellipsis and an
/// "and N more." footer. Detailed stats (removed/renamed breakdown) are
/// deferred to a future verbose mode; `--json` carries the full picture.
const MAX_LISTED: usize = 10;

fn render_survey(result: &DetectResult, has_index: bool) {
    let pending = result.pending();

    if !has_index {
        // First run: no database yet.
        println!("No existing lode found.");
        if pending == 0 {
            println!();
            println!("The lode is empty — nothing to mine.");
            return;
        }
        println!();
        println!("The lode reveals:");
        println!();
        print_changes(result);
        println!();
        println!("{pending} files await mining.");
        println!();
        println!("Run `lode mine` to start mining.");
        return;
    }

    if pending == 0 {
        // Index present, nothing changed.
        println!("No new findings in this lode.");
        return;
    }

    // Index present, changes detected.
    println!("New findings since last mine:");
    println!();
    print_changes(result);
    println!();
    println!("{pending} files await mining.");
    println!();
    println!("Run `lode mine` to continue mining.");
}

/// Print the pending change list, truncated to [`MAX_LISTED`] entries.
fn print_changes(result: &DetectResult) {
    let total = result.changes.len();
    let shown = total.min(MAX_LISTED);

    for change in result.changes.iter().take(shown) {
        match change {
            Change::Added(snap) => println!("  + {}", snap.path.as_str()),
            Change::Modified { old, .. } => println!("  ~ {}", old.path.as_str()),
            Change::Removed(record) => println!("  - {}", record.path.as_str()),
            Change::Renamed { from, to } => {
                println!("  > {} -> {}", from.as_str(), to.as_str())
            }
        }
    }

    if total > shown {
        println!("  ...");
        println!("  and {} more.", total - shown);
    }
}

/// Render the detailed (`extended`) survey report.
///
/// Same narrative scenarios as `compact`, but with a `Summary` stats block
/// up front, an untruncated change list grouped by status (each with an
/// inline subtotal), `size`/`mtime` detail on modified files, and a
/// skipped count.
fn render_survey_extended(result: &DetectResult, has_index: bool) {
    let pending = result.pending();

    if !has_index && pending == 0 {
        // First run, empty workspace.
        println!("No existing lode found.");
        println!();
        println!("The lode is empty — nothing to mine.");
        return;
    }

    if !has_index {
        println!("No existing lode found.");
    } else if pending == 0 {
        println!("No new findings in this lode.");
    } else {
        println!("New findings since last mine.");
    }

    println!();
    print_summary(result);

    if pending > 0 {
        println!();
        print_changes_extended(result);
        println!();
        println!("{pending} files await mining.");
        println!();
        if has_index {
            println!("Run `lode mine` to continue mining.");
        } else {
            println!("Run `lode mine` to start mining.");
        }
    }
}

/// Print the full change list: grouped by status with inline subtotals.
///
/// Unlike [`print_changes`], this is never truncated. Modified entries show
/// the old → new `size` and `mtime` so the reason for the change is visible.
fn print_changes_extended(result: &DetectResult) {
    println!("Changes:");
    println!();
    print_group(
        "New",
        result.added_count(),
        result.changes.iter().filter_map(|c| match c {
            Change::Added(snap) => Some(format!("+ {}", snap.path.as_str())),
            _ => None,
        }),
    );
    print_group(
        "Modified",
        result.modified_count(),
        result.changes.iter().filter_map(|c| match c {
            Change::Modified { old, new } => Some(format!(
                "~ {}  ({} -> {} bytes, {} -> {})",
                old.path.as_str(),
                old.size,
                new.size,
                format_unix_ts(old.mtime),
                format_unix_ts(new.mtime),
            )),
            _ => None,
        }),
    );
    print_group(
        "Missing",
        result.removed_count(),
        result.changes.iter().filter_map(|c| match c {
            Change::Removed(record) => Some(format!("- {}", record.path.as_str())),
            _ => None,
        }),
    );
    print_group(
        "Renamed",
        result.renamed_count(),
        result.changes.iter().filter_map(|c| match c {
            Change::Renamed { from, to } => Some(format!("> {} -> {}", from.as_str(), to.as_str())),
            _ => None,
        }),
    );
}

/// Print the per-status summary block.
///
/// Change-status counts (`New`/`Modified`/`Missing`/`Renamed`) are aligned
/// together; `Skipped` is deliberately kept OUT of that block and rendered
/// afterwards as a natural-language clause, signalling visually that it is
/// not a change status but a non-actionable tally of unsupported files.
fn print_summary(result: &DetectResult) {
    println!("Summary:");
    println!();
    println!("  {:<9} {}", "New:", format_count(result.added_count()));
    println!(
        "  {:<9} {}",
        "Modified:",
        format_count(result.modified_count())
    );
    println!(
        "  {:<9} {}",
        "Missing:",
        format_count(result.removed_count())
    );
    println!(
        "  {:<9} {}",
        "Renamed:",
        format_count(result.renamed_count())
    );
    if !result.skipped.is_empty() {
        println!();
        println!(
            "Skipped {} files because they are unsupported.",
            format_count(result.skipped.len())
        );
    }
}

/// Format a count with thousands separators, e.g. `7371` → `7,371`.
fn format_count(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Print a status group with an inline subtotal, skipping empty groups.
fn print_group(label: &str, count: usize, lines: impl Iterator<Item = String>) {
    if count == 0 {
        return;
    }
    println!("  {label} ({count}):");
    for line in lines {
        println!("    {line}");
    }
}

/// Format a Unix timestamp (seconds) as `YYYY-MM-DD HH:MM:SS` (UTC).
fn format_unix_ts(ts: f64) -> String {
    let total = ts as i64;
    let days = total.div_euclid(86_400);
    let secs = total.rem_euclid(86_400);
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02}")
}

/// Convert days since 1970-01-01 to `(year, month, day)`.
///
/// Howard Hinnant's `civil_from_days` algorithm; avoids a chrono dependency
/// for the single timestamp format the CLI needs.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}
