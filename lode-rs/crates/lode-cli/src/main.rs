use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use lode_core::index::records::FileRecord;
use lode_core::index::store::Store;
use lode_core::ingestion::pipeline::{Change, DetectResult, classify, detect_changes};
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
/// `compact` is the default narrative; `extended` adds detail; `json` is
/// the machine-readable form. Views are orthogonal to `--log-level` (stderr).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum View {
    /// Compact narrative (default).
    Compact,
    /// Detailed narrative with stats and full lists.
    Extended,
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
    Mine,
    /// Search the store (alias: search).
    Prospect,
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
        Command::Mine => todo_command("mine"),
        Command::Prospect => todo_command("prospect"),
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

/// Run the survey command: detect workspace changes and report stale files.
///
/// Detection only — never touches the embedder or creates a database. With
/// no index yet, it classifies the workspace against an empty snapshot
/// (every supported file is `new`), so a user can see what `mine` would
/// index before running it.
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
            Ok(store) => match detect_changes(&store, workspace, &[]) {
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
        View::Json => emit_json(&result),
    }
    0
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
        if i > 0 && (digits.len() - i) % 3 == 0 {
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
