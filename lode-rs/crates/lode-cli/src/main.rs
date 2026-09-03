use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
    #[arg(short = 'C', long, default_value = ".")]
    workspace: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report ingestion / index status (alias: status).
    #[command(alias = "status")]
    Survey {
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
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
    match cli.command {
        Command::Survey { json } => survey(&cli.workspace, json),
        Command::Mine => todo_command("mine"),
        Command::Prospect => todo_command("prospect"),
        Command::Dig => todo_command("dig"),
        Command::Assay => todo_command("assay"),
        Command::Config => todo_command("config"),
    }
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
fn survey(workspace: &std::path::Path, as_json: bool) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("survey", workspace) {
        return code;
    }

    let db_path = workspace.join(INDEX_DB_RELATIVE);
    let has_index = db_path.is_file();

    let result = if has_index {
        match Store::open_existing(&db_path) {
            Ok(store) => match detect_changes(&store, workspace, &[]) {
                Ok(result) => result,
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
        let indexed: HashMap<WorkspacePath, FileRecord> = HashMap::new();
        classify(&indexed, workspace, &[])
    };

    if as_json {
        emit_json(workspace, &result);
    } else {
        render_survey(&result, has_index);
    }
    0
}

/// Emit a JSON survey payload.
fn emit_json(workspace: &std::path::Path, result: &DetectResult) {
    let mut new_paths = Vec::new();
    let mut changed_paths = Vec::new();
    let mut missing_paths = Vec::new();
    let mut renamed_pairs = Vec::new();
    let mut unchanged_paths = Vec::new();

    for change in &result.changes {
        match change {
            Change::Added(snap) => new_paths.push(snap.path.as_str()),
            Change::Modified { old, .. } => changed_paths.push(old.path.as_str()),
            Change::Removed(record) => missing_paths.push(record.path.as_str()),
            Change::Renamed { from, to } => renamed_pairs.push((from.as_str(), to.as_str())),
        }
    }
    for path in &result.unchanged {
        unchanged_paths.push(path.as_str());
    }

    let payload = serde_json::json!({
        "ok": true,
        "command": "survey",
        "workspace": workspace.to_string_lossy(),
        "summary": {
            "unchanged": unchanged_paths.len(),
            "new": new_paths.len(),
            "changed": changed_paths.len(),
            "missing": missing_paths.len(),
            "renamed": renamed_pairs.len(),
            "skipped": result.skipped.len(),
            "pending": result.pending(),
        },
        "paths": {
            "new": new_paths,
            "changed": changed_paths,
            "missing": missing_paths,
            "renamed": renamed_pairs.iter().map(|(f, t)| serde_json::json!({"from": f, "to": t})).collect::<Vec<_>>(),
            "unchanged": unchanged_paths,
        },
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
