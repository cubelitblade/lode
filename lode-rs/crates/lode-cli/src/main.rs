use std::collections::HashMap;
use std::fmt::Arguments;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use terminal_size::{Width, terminal_size};

use lode_core::config::layered::{self, load_settings_for};
use lode_core::config::schema;
use lode_core::config::{OutputPalette, Settings, build_embedder};
use lode_core::index::records::{ChunkLookup, ChunkWithRefs, FileRecord};
use lode_core::index::store::Store;
use lode_core::ingestion::pipeline::{
    Change, DetectResult, SyncSummary, classify, detect_changes, sync,
};
use lode_core::ingestion::split::RecursiveSegmentSplitter;
use lode_core::relpath::WorkspacePath;

mod markdown_render;
mod render;

use render::{Intent, RenderOptions};

const HELP_TEMPLATE: &str = "\
Lode {version}

{about}

{usage-heading} {usage}

{all-args}
";

/// Process exit status contract for commands that get past clap parsing.
const EXIT_OK: u8 = 0;
const EXIT_ERROR: u8 = 1;
const EXIT_PARTIAL_FAILURE: u8 = 3;

/// Stable machine-readable error identifiers. Keep these independent from
/// infrastructure error text so JSON consumers can branch safely.
mod error_code {
    pub const NOT_IMPLEMENTED: &str = "not_implemented";
    pub const WORKSPACE_MISSING: &str = "workspace_missing";
    pub const WORKSPACE_NOT_DIRECTORY: &str = "workspace_not_directory";
    pub const CONFIG: &str = "config_error";
    pub const EMBEDDING: &str = "embedding_error";
    pub const CHUNKING: &str = "chunking_error";
    pub const DIMENSION: &str = "dimension_error";
    pub const MODEL: &str = "model_error";
    pub const RESET: &str = "reset_error";
    pub const MINING: &str = "mining_error";
    pub const SCAN: &str = "scan_error";
    pub const INDEX_OPEN: &str = "index_open_error";
    pub const INDEX_CREATE: &str = "index_create_error";
    pub const INDEX_READ: &str = "index_read_error";
    pub const SEARCH: &str = "search_error";
    pub const INVALID_TOP_K: &str = "invalid_top_k";
    pub const INVALID_QUERY: &str = "invalid_query";
    pub const NO_INDEX: &str = "no_index";
    pub const INVALID_DIGEST: &str = "invalid_digest";
    pub const NOT_FOUND: &str = "not_found";
    pub const AMBIGUOUS: &str = "ambiguous";
}

static STDOUT_ERROR: OnceLock<Mutex<Option<io::ErrorKind>>> = OnceLock::new();
static STDERR_ERROR: OnceLock<Mutex<Option<io::ErrorKind>>> = OnceLock::new();

fn write_line(
    stream: &mut impl Write,
    state: &OnceLock<Mutex<Option<io::ErrorKind>>>,
    args: Arguments<'_>,
) {
    let error = state.get_or_init(|| Mutex::new(None));
    if error
        .lock()
        .expect("output error state is not poisoned")
        .is_some()
    {
        return;
    }
    if let Err(write_error) = writeln!(stream, "{args}") {
        *error.lock().expect("output error state is not poisoned") = Some(write_error.kind());
    }
}

fn write_stdout(args: Arguments<'_>) {
    write_line(&mut io::stdout().lock(), &STDOUT_ERROR, args);
}

fn write_stderr(args: Arguments<'_>) {
    write_line(&mut io::stderr().lock(), &STDERR_ERROR, args);
}

macro_rules! println {
    () => { $crate::write_stdout(format_args!("")) };
    ($($arg:tt)*) => { $crate::write_stdout(format_args!($($arg)*)) };
}

macro_rules! eprintln {
    () => { $crate::write_stderr(format_args!("")) };
    ($($arg:tt)*) => { $crate::write_stderr(format_args!($($arg)*)) };
}

fn output_errors() -> (Option<io::ErrorKind>, Option<io::ErrorKind>) {
    let stdout = STDOUT_ERROR
        .get()
        .and_then(|error| *error.lock().expect("output error state is not poisoned"));
    let stderr = STDERR_ERROR
        .get()
        .and_then(|error| *error.lock().expect("output error state is not poisoned"));
    (stdout, stderr)
}

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
    /// Stable TSV records, suitable for scripts and line-oriented tools.
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

/// CLI spelling for the configured output palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PaletteArg {
    Ansi,
    #[value(name = "accessible_tol")]
    AccessibleTol,
    #[value(name = "accessible_okabe_ito")]
    AccessibleOkabeIto,
}

impl PaletteArg {
    const fn into_core(self) -> OutputPalette {
        match self {
            Self::Ansi => OutputPalette::Ansi,
            Self::AccessibleTol => OutputPalette::AccessibleTol,
            Self::AccessibleOkabeIto => OutputPalette::AccessibleOkabeIto,
        }
    }
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
    color = clap::ColorChoice::Never,
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

    /// Colour palette for human-readable output.
    #[arg(long, value_enum, global = true)]
    palette: Option<PaletteArg>,

    /// Disable colour for this invocation.
    #[arg(long, global = true)]
    no_color: bool,

    /// Disable progress indicators for this invocation.
    #[arg(long, global = true)]
    no_progress: bool,

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
    #[command(alias = "get")]
    Dig {
        /// Chunk digest or hexadecimal prefix.
        digest: String,
        /// Number of adjacent chunks to include on each side.
        #[arg(long, default_value_t = 0)]
        radius: u32,
    },
    /// Analyze the store: why | how.
    Assay,
    /// Show or edit configuration.
    Config {
        #[command(subcommand)]
        command: Option<ConfigCommand>,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Show the merged effective configuration.
    Show,
    /// Read one configuration key.
    Get {
        /// Canonical dotted configuration key.
        key: String,
        /// Read only the explicit value from one layer.
        #[arg(long, value_enum)]
        scope: Option<ConfigScopeArg>,
    },
    /// Write one configuration key.
    Set {
        /// Canonical dotted configuration key.
        key: String,
        /// Value parsed according to the key's type.
        #[arg(allow_negative_numbers = true)]
        value: String,
        /// Layer to edit.
        #[arg(long, value_enum, default_value_t = ConfigScopeArg::Workspace)]
        scope: ConfigScopeArg,
    },
    /// Remove one explicit configuration key.
    Unset {
        /// Canonical dotted configuration key.
        key: String,
        /// Layer to edit.
        #[arg(long, value_enum, default_value_t = ConfigScopeArg::Workspace)]
        scope: ConfigScopeArg,
    },
    /// Print the path of a configuration layer.
    Path {
        /// Layer whose write path should be shown.
        #[arg(long, value_enum, default_value_t = ConfigScopeArg::Workspace)]
        scope: ConfigScopeArg,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ConfigScopeArg {
    User,
    Workspace,
}

impl ConfigScopeArg {
    const fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Workspace => "workspace",
        }
    }
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let code = dispatch(cli);
    let (stdout_error, stderr_error) = output_errors();
    if stdout_error
        .into_iter()
        .chain(stderr_error)
        .any(|kind| kind != io::ErrorKind::BrokenPipe)
    {
        return std::process::ExitCode::from(EXIT_ERROR);
    }
    if stdout_error.is_some() || stderr_error.is_some() {
        // A closed consumer is a successful producer shutdown only when the
        // command itself succeeded. Preserve command failures and partial
        // failures so scripts can still act on their exit status.
        return std::process::ExitCode::from(code);
    }
    std::process::ExitCode::from(code)
}

/// Route a parsed subcommand to its implementation.
fn dispatch(cli: Cli) -> u8 {
    let log_level = resolve_log_level(cli.verbose, cli.log_level);
    init_logging(log_level);
    let cli_palette = cli.palette.map(PaletteArg::into_core);
    render::set_output(
        cli_palette.unwrap_or(OutputPalette::Ansi),
        cli.no_color.then_some(true),
    );
    match cli.command {
        Command::Survey => survey(&cli.workspace, cli.view, cli.palette, cli.no_color),
        Command::Mine { from_scratch } => mine(
            &cli.workspace,
            cli.view,
            from_scratch,
            cli.palette,
            cli.no_color,
            cli.no_progress,
        ),
        Command::Prospect { query, top_k } => prospect(
            &cli.workspace,
            query,
            top_k,
            cli.view,
            cli.palette,
            cli.no_color,
        ),
        Command::Dig { digest, radius } => dig(
            &cli.workspace,
            digest,
            radius,
            cli.view,
            cli.palette,
            cli.no_color,
        ),
        Command::Assay => todo_command("assay", cli.view),
        Command::Config { command } => config_command(
            &cli.workspace,
            command.unwrap_or(ConfigCommand::Show),
            cli.view,
            cli.palette,
            cli.no_color,
        ),
    }
}

/// Load project settings after applying this invocation's output flags.
fn configure_output(
    workspace: &std::path::Path,
    command: &str,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> Result<Settings, u8> {
    match load_settings_for(workspace) {
        Ok(mut settings) => {
            if let Some(palette) = palette {
                settings.app.output.palette = palette.into_core();
            }
            if no_color {
                settings.app.output.no_color = Some(true);
            }
            render::set_output(settings.app.output.palette, settings.app.output.no_color);
            Ok(settings)
        }
        Err(error) => Err(ui_msg::die(
            view,
            command,
            error_code::CONFIG,
            &format!("Could not load configuration: {error}"),
            Some("Check your lode.toml and environment."),
        )),
    }
}

fn configure_unloaded_output(palette: Option<PaletteArg>, no_color: bool) {
    render::set_output(
        palette
            .map(PaletteArg::into_core)
            .unwrap_or(OutputPalette::Ansi),
        no_color.then_some(true),
    );
}

fn config_command(
    workspace: &std::path::Path,
    command: ConfigCommand,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("config", workspace, view) {
        return code;
    }
    match command {
        ConfigCommand::Show => {
            let settings = match configure_output(workspace, "config", view, palette, no_color) {
                Ok(settings) => settings,
                Err(code) => return code,
            };
            let json = match schema::redacted_json(&settings) {
                Ok(value) => value,
                Err(error) => return config_error(view, &error),
            };
            match view {
                View::Compact | View::Extended => match schema::redacted_toml(&settings) {
                    Ok(value) => println!("{value}"),
                    Err(error) => return config_error(view, &error),
                },
                View::Table => render_config_table(&json),
                View::Json => println!("{}", serde_json::to_string_pretty(&json).unwrap()),
            }
            EXIT_OK
        }
        ConfigCommand::Get { key, scope } => {
            config_get(workspace, &key, scope, view, palette, no_color)
        }
        ConfigCommand::Path { scope } => {
            configure_unloaded_output(palette, no_color);
            let path = match config_scope_path(workspace, scope) {
                Ok(path) => path,
                Err(error) => return config_error(view, &error),
            };
            match view {
                View::Compact | View::Extended => println!("{}", path.display()),
                View::Table => {
                    print_tsv(&["KEY", "VALUE", "PATH"]);
                    let path = path.display().to_string();
                    print_tsv(&[scope.label(), "", &path]);
                }
                View::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({"path": path})).unwrap()
                ),
            }
            EXIT_OK
        }
        ConfigCommand::Set { key, value, scope } => {
            config_set(workspace, &key, &value, scope, view, palette, no_color)
        }
        ConfigCommand::Unset { key, scope } => {
            config_unset(workspace, &key, scope, view, palette, no_color)
        }
    }
}

fn config_error(view: View, error: &impl std::fmt::Display) -> u8 {
    ui_msg::die(
        view,
        "config",
        error_code::CONFIG,
        &error.to_string(),
        Some("Check the configuration key, value, and target file."),
    )
}

fn config_scope_path(
    workspace: &std::path::Path,
    scope: ConfigScopeArg,
) -> lode_core::Result<PathBuf> {
    match scope {
        ConfigScopeArg::User => layered::user_config_path(),
        ConfigScopeArg::Workspace => Ok(layered::workspace_config_path(workspace)),
    }
}

fn editable_config_key(key: &str) -> lode_core::Result<&'static schema::ConfigKey> {
    schema::find_key(key)
        .filter(|key| key.editable)
        .ok_or_else(|| schema::unknown_key(key))
}

fn config_get(
    workspace: &std::path::Path,
    key: &str,
    scope: Option<ConfigScopeArg>,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    let spec = match schema::find_key(key) {
        Some(spec) => spec,
        None => return config_error(view, &schema::unknown_key(key)),
    };
    let (value, path) = match scope {
        Some(scope) => {
            configure_unloaded_output(palette, no_color);
            let path = match config_scope_path(workspace, scope) {
                Ok(path) => path,
                Err(error) => return config_error(view, &error),
            };
            let value = match layered::get_explicit_value(&path, key) {
                Ok(Some(value)) => serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
                Ok(None) => {
                    return config_error(
                        view,
                        &lode_core::Error::Config(format!(
                            "configuration key `{key}` is not set in the {} scope",
                            scope.label()
                        )),
                    );
                }
                Err(error) => return config_error(view, &error),
            };
            (redact_config_value(spec, value), path.display().to_string())
        }
        None => {
            let settings = match configure_output(workspace, "config", view, palette, no_color) {
                Ok(settings) => settings,
                Err(code) => return code,
            };
            let settings = match schema::redacted_json(&settings) {
                Ok(value) => value,
                Err(error) => return config_error(view, &error),
            };
            let Some(value) = schema::json_path(&settings, key).cloned() else {
                return config_error(
                    view,
                    &lode_core::Error::Config(format!("configuration key `{key}` is unavailable")),
                );
            };
            (value, "effective".to_string())
        }
    };
    render_config_get(view, key, &value, &path)
}

fn redact_config_value(spec: &schema::ConfigKey, value: serde_json::Value) -> serde_json::Value {
    if spec.sensitive && !value.is_null() {
        serde_json::Value::String("[redacted]".to_string())
    } else {
        value
    }
}

fn render_config_get(view: View, key: &str, value: &serde_json::Value, path: &str) -> u8 {
    match view {
        View::Compact => println!("{}", display_json_value(value)),
        View::Extended => println!("{key} = {} ({path})", display_json_value(value)),
        View::Table => {
            print_tsv(&["KEY", "VALUE", "PATH"]);
            let value = display_json_value(value);
            print_tsv(&[key, &value, path]);
        }
        View::Json => println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "key": key,
                "value": value,
            }))
            .unwrap()
        ),
    }
    EXIT_OK
}

fn config_set(
    workspace: &std::path::Path,
    key: &str,
    raw_value: &str,
    scope: ConfigScopeArg,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    let spec = match editable_config_key(key) {
        Ok(spec) => spec,
        Err(error) => return config_error(view, &error),
    };
    let value = match schema::parse_value(key, raw_value) {
        Ok(value) => value,
        Err(error) => return config_error(view, &error),
    };
    configure_unloaded_output(palette, no_color);
    let path = match config_scope_path(workspace, scope) {
        Ok(path) => path,
        Err(error) => return config_error(view, &error),
    };
    let changed = match layered::set_value(&path, key, value.clone()) {
        Ok(changed) => changed,
        Err(error) => return config_error(view, &error),
    };
    render_config_edit(view, "set", key, &value, &path, changed, spec)
}

fn config_unset(
    workspace: &std::path::Path,
    key: &str,
    scope: ConfigScopeArg,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    let spec = match editable_config_key(key) {
        Ok(spec) => spec,
        Err(error) => return config_error(view, &error),
    };
    configure_unloaded_output(palette, no_color);
    let path = match config_scope_path(workspace, scope) {
        Ok(path) => path,
        Err(error) => return config_error(view, &error),
    };
    let changed = match layered::unset_value(&path, key) {
        Ok(changed) => changed,
        Err(error) => return config_error(view, &error),
    };
    render_config_edit(
        view,
        "unset",
        key,
        &toml::Value::String(String::new()),
        &path,
        changed,
        spec,
    )
}

fn render_config_edit(
    view: View,
    operation: &str,
    key: &str,
    value: &toml::Value,
    path: &std::path::Path,
    changed: bool,
    spec: &schema::ConfigKey,
) -> u8 {
    let json_value = if operation == "unset" {
        serde_json::Value::Null
    } else {
        redact_config_value(
            spec,
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        )
    };
    match view {
        View::Compact => println!("{operation} {key}"),
        View::Extended => println!(
            "{operation} {key} = {} in {}",
            display_json_value(&json_value),
            path.display()
        ),
        View::Table => {
            print_tsv(&["KEY", "VALUE", "PATH"]);
            let value = display_json_value(&json_value);
            let path = path.display().to_string();
            print_tsv(&[key, &value, &path]);
        }
        View::Json => println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "key": key,
                "path": path,
                "changed": changed,
            }))
            .unwrap()
        ),
    }
    EXIT_OK
}

fn render_config_table(config: &serde_json::Value) {
    print_tsv(&["KEY", "VALUE", "PATH"]);
    for spec in schema::CONFIG_KEYS {
        if let Some(value) = schema::json_path(config, spec.path) {
            let value = display_json_value(value);
            print_tsv(&[spec.path, &value, "effective"]);
        }
    }
}

fn display_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Null => "null".to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
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
        .write_style(env_logger::fmt::WriteStyle::Never)
        .init();
}

/// Not-yet-implemented command: print a TODO on stderr and exit non-zero.
fn todo_command(name: &str, view: View) -> u8 {
    ui_msg::die(
        view,
        name,
        error_code::NOT_IMPLEMENTED,
        &format!("`{name}` is not implemented yet"),
        None,
    )
}

/// Centralised user-facing error text, mirroring the Python `messages.py`
/// convention: every surfaced error pairs a headline with an actionable hint.
///
/// Kept deliberately narrow — only shapes the current commands need. Expand
/// it (rather than scattering ad-hoc `eprintln!`) when a new shape arises.
mod ui_msg {
    use super::{EXIT_ERROR, Intent, RenderOptions, View, error_code};

    /// Write one stable machine-readable error object, or its human form.
    pub fn die(view: View, command: &str, code: &str, error: &str, hint: Option<&str>) -> u8 {
        die_with_extra(view, command, code, error, hint, None)
    }

    /// Write an error while retaining command-specific structured fields.
    pub fn die_with_extra(
        view: View,
        command: &str,
        code: &str,
        error: &str,
        hint: Option<&str>,
        extra: Option<serde_json::Value>,
    ) -> u8 {
        if view == View::Json {
            let mut payload = serde_json::json!({
                "code": code,
                "message": error,
                "hint": hint,
            });
            if let Some(extra) = extra
                && let (Some(object), Some(extra_object)) =
                    (payload.as_object_mut(), extra.as_object())
            {
                object.extend(extra_object.clone());
            }
            println!("{}", serde_json::to_string_pretty(&payload).unwrap());
            return EXIT_ERROR;
        }

        let options = RenderOptions::stderr();
        eprintln!(
            "{}",
            options.paint(Intent::Error, &format!("[{command}] {error}"))
        );
        if let Some(hint) = hint {
            eprintln!(
                "{}",
                options.paint(Intent::Info, &format!("       Hint: {hint}"))
            );
        }
        EXIT_ERROR
    }

    /// Validate the workspace path: it must exist and be a directory.
    ///
    /// Mirrors the Python app-level callback (`exists=True`,
    /// `file_okay=False`, `dir_okay=True`).
    pub fn bad_workspace(command: &str, ws: &std::path::Path, view: View) -> Option<u8> {
        if !ws.exists() {
            return Some(die(
                view,
                command,
                error_code::WORKSPACE_MISSING,
                &format!("Workspace '{}' does not exist.", ws.display()),
                Some("Point `--workspace/-C` at an existing directory."),
            ));
        }
        if !ws.is_dir() {
            return Some(die(
                view,
                command,
                error_code::WORKSPACE_NOT_DIRECTORY,
                &format!("'{0}' is not a directory.", ws.display()),
                Some("Provide a directory, not a regular file."),
            ));
        }
        None
    }
}

/// The relative path of the index database within the workspace.
const INDEX_DB_RELATIVE: &str = ".lode/index.db";

/// Run the dig command: fetch one chunk and an optional same-section window.
fn dig(
    workspace: &std::path::Path,
    input: String,
    radius: u32,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("dig", workspace, view) {
        return code;
    }
    let settings = match configure_output(workspace, "dig", view, palette, no_color) {
        Ok(settings) => settings,
        Err(code) => return code,
    };

    let db_path = workspace.join(INDEX_DB_RELATIVE);
    if !db_path.is_file() {
        let text = lode_core::messages::require("no_index");
        let index_path = db_path.display().to_string();
        let error = lode_core::messages::format(text.error, &[("index_path", index_path)]);
        return dig_error_with_text(view, &error, text.hint, error_code::NO_INDEX, &input, &[]);
    }

    let mut store = match Store::open_existing(&db_path) {
        Ok(store) => store,
        Err(e) => {
            return ui_msg::die(
                view,
                "dig",
                error_code::INDEX_OPEN,
                &format!("Cannot open the lode index: {e}"),
                Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
            );
        }
    };
    let ignore_files = settings
        .app
        .ignore
        .sources
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if let Err(e) = detect_changes(&mut store, workspace, &ignore_files) {
        return ui_msg::die(
            view,
            "dig",
            error_code::SCAN,
            &format!("Could not finish scanning the workspace: {e}"),
            Some("Fix the underlying error and rerun `lode dig`."),
        );
    }

    let token = match normalize_digest(&input) {
        Some(token) => token,
        None => return dig_error(view, error_code::INVALID_DIGEST, &input, &[]),
    };

    let lookup = match store.get_chunk_window(&token, radius) {
        Ok(lookup) => lookup,
        Err(e) => {
            return ui_msg::die(
                view,
                "dig",
                error_code::INDEX_READ,
                &format!("Could not read the lode index: {e}"),
                Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
            );
        }
    };
    let (target, window) = match lookup {
        ChunkLookup::NotFound => return dig_error(view, error_code::NOT_FOUND, &input, &[]),
        ChunkLookup::Ambiguous(candidates) => {
            return dig_error(view, error_code::AMBIGUOUS, &input, &candidates);
        }
        ChunkLookup::Found(window) => (window.target, window.chunks),
    };

    match view {
        View::Compact => render_dig(&target, &window),
        View::Extended => render_dig_extended(&input, &target, &window, radius),
        View::Table => render_dig_table(&window, &target.digest, target.seq),
        View::Json => emit_dig_json(&target, &window, radius),
    }
    0
}

/// Normalize and validate a user-supplied digest or hexadecimal prefix.
fn normalize_digest(input: &str) -> Option<String> {
    let token = input.trim().strip_prefix("blake3:").unwrap_or(input.trim());
    let token = token.to_ascii_lowercase();
    if !token.is_empty() && token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Some(token)
    } else {
        None
    }
}

fn dig_error(view: View, code: &str, input: &str, candidates: &[ChunkWithRefs]) -> u8 {
    let text = lode_core::messages::require(code);
    let count = candidates.len();
    let error = lode_core::messages::format(
        text.error,
        &[("digest", input.to_string()), ("count", count.to_string())],
    );
    dig_error_with_text(view, &error, text.hint, code, input, candidates)
}

fn dig_error_with_text(
    view: View,
    error: &str,
    hint: Option<&str>,
    code: &str,
    _input: &str,
    candidates: &[ChunkWithRefs],
) -> u8 {
    let extra = if code == error_code::AMBIGUOUS {
        Some(serde_json::json!({
            "candidates": candidates.iter().map(dig_chunk_json_without_text).collect::<Vec<_>>(),
        }))
    } else {
        None
    };
    let result = ui_msg::die_with_extra(view, "dig", code, error, hint, extra);
    if view != View::Json && code == error_code::AMBIGUOUS {
        for candidate in candidates {
            eprintln!(
                "  #{} {}",
                short_id(&candidate.digest),
                dig_source_line(candidate)
            );
        }
    }
    result
}

fn dig_source_line(chunk: &ChunkWithRefs) -> String {
    let primary = chunk.primary();
    let mut source = primary.path.to_native().to_string_lossy().into_owned();
    if chunk.refs.len() > 1 {
        source += &format!(" (+{} more)", chunk.refs.len() - 1);
    }
    if !chunk.heading.is_empty() {
        source += &format!(" > {}", chunk.heading);
    }
    if let Some(page) = chunk.page {
        source += &format!(" (p.{page})");
    }
    if primary.status == lode_core::index::records::FileStatus::Stale {
        source += " [stale]";
    }
    source
}

fn dig_chunk_json(chunk: &ChunkWithRefs) -> serde_json::Value {
    let mut value = dig_chunk_json_without_text(chunk);
    value["text"] = serde_json::Value::String(chunk.text.clone());
    value
}

fn dig_chunk_json_without_text(chunk: &ChunkWithRefs) -> serde_json::Value {
    serde_json::json!({
        "digest": chunk.digest,
        "paths": chunk.refs.iter().map(|reference| serde_json::json!({
            "path": reference.path.as_str(),
            "state": reference.status.to_string(),
        })).collect::<Vec<_>>(),
        "heading": chunk.heading,
        "page": chunk.page,
        "seq": chunk.seq,
    })
}

fn emit_dig_json(target: &ChunkWithRefs, window: &[ChunkWithRefs], radius: u32) {
    let payload = serde_json::json!({
        "digest": target.digest,
        "window": {
            "center_seq": target.seq,
            "radius": radius,
            "chunks": window.iter().map(dig_chunk_json).collect::<Vec<_>>(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

fn render_dig_header(digest: &str, radius: u32, options: RenderOptions) {
    let label = short_id(digest);
    if radius == 0 {
        render::print_line(options, Intent::Info, &format!("Dug {label}"));
    } else {
        render::print_line(
            options,
            Intent::Info,
            &format!("Dug {label} with radius {radius}."),
        );
    }
}

fn render_dig(target: &ChunkWithRefs, window: &[ChunkWithRefs]) {
    let options = RenderOptions::stdout();
    if render_dig_compact_body(target, window, options) {
        return;
    }
    render_dig_compact_cards_body(target, window, options);
}

/// Render the compact metadata shared by the merged and card fallbacks.
///
/// Keeping the source and target marker above one divider makes the first
/// line of the body unambiguous even when colour is disabled or output is
/// being read from a narrow terminal.
fn render_dig_compact_metadata(
    target: &ChunkWithRefs,
    window: &[ChunkWithRefs],
    options: RenderOptions,
) {
    let source_intent = if chunk_stale(target) {
        Intent::Warning
    } else {
        Intent::Info
    };
    render::print_line(options, source_intent, &dig_source_line(target));
    let target_label = target.seq.map_or_else(
        || short_id(&target.digest).to_string(),
        |seq| format!("chunk {seq}"),
    );
    render::print_line(options, Intent::Muted, &format!("Target · {target_label}"));

    let (before, after) = dig_context_counts(target, window);
    if before + after > 0 {
        let context = format!("Context · {} before, {} after", before, after);
        render::print_line(options, Intent::Muted, &context);
    }
    render::print_line(options, Intent::Muted, &compact_divider());
}

fn compact_divider() -> String {
    let terminal_width = terminal_size()
        .map(|(Width(width), _)| width as usize)
        .unwrap_or(80);
    let divider_width = terminal_width.saturating_sub(2).clamp(1, 80);
    "─".repeat(divider_width)
}

fn dig_context_counts(target: &ChunkWithRefs, window: &[ChunkWithRefs]) -> (usize, usize) {
    let Some(target_index) = window
        .iter()
        .position(|chunk| chunk.digest == target.digest && chunk.seq == target.seq)
    else {
        return (0, 0);
    };
    (target_index, window.len().saturating_sub(target_index + 1))
}

/// Render a compact fallback when source spans are unavailable or the
/// fragment must be rendered as Markdown. The target metadata is shown once;
/// each chunk then gets a small positional label so the window remains easy
/// to scan without relying on a card border or terminal colour.
fn render_dig_compact_cards_body(
    target: &ChunkWithRefs,
    window: &[ChunkWithRefs],
    options: RenderOptions,
) {
    render_dig_compact_metadata(target, window, options);
    let target_source = dig_source_line(target);
    let target_index = window
        .iter()
        .position(|chunk| chunk.digest == target.digest && chunk.seq == target.seq);
    for (index, chunk) in window.iter().enumerate() {
        let is_target = chunk.digest == target.digest && chunk.seq == target.seq;
        let label = if is_target {
            "Target"
        } else if target_index.is_some_and(|target_index| index < target_index) {
            "Before"
        } else {
            "After"
        };
        let chunk_label = chunk.seq.map_or_else(
            || short_id(&chunk.digest).to_string(),
            |seq| format!("chunk {seq}"),
        );
        println!();
        render::print_line(
            options,
            if is_target {
                Intent::Info
            } else {
                Intent::Muted
            },
            &format!("{label} · {chunk_label}"),
        );
        let source = dig_source_line(chunk);
        if source != target_source {
            let source_intent = if chunk_stale(chunk) {
                Intent::Warning
            } else {
                Intent::Muted
            };
            render::print_line(options, source_intent, &format!("  {source}"));
        }
        let rendered = markdown_render::render_fragment(
            &chunk.text,
            is_markdown_content(&chunk.extractor),
            options,
        );
        print_indented("  ", &rendered);
    }
    println!();
    render::print_line(options, Intent::Muted, short_id(&target.digest));
}

/// Render the legacy one-card-per-chunk layout.
fn render_dig_cards_body(target: &ChunkWithRefs, window: &[ChunkWithRefs], options: RenderOptions) {
    for chunk in window {
        let center = chunk.seq == target.seq;
        let title = match chunk.seq {
            Some(seq) => seq.to_string(),
            None => short_id(&chunk.digest).to_string(),
        };
        println!();
        render::print_line(
            options,
            Intent::Info,
            &format!("{}{}", title, if center { " · center" } else { "" }),
        );
        render::print_line(
            options,
            Intent::Muted,
            &format!("  {}", dig_source_line(chunk)),
        );
        let rendered = markdown_render::render_fragment(
            &chunk.text,
            is_markdown_content(&chunk.extractor),
            options,
        );
        print_indented("  ", &rendered);
        render::print_line(
            options,
            Intent::Muted,
            &format!("  {}", short_id(&chunk.digest)),
        );
    }
}

/// Render one overlap-deduplicated context block when every chunk has a
/// valid span in the same segment. Returning `false` keeps the old card
/// layout as the safe fallback for synthetic or legacy rows.
fn render_dig_compact_body(
    target: &ChunkWithRefs,
    window: &[ChunkWithRefs],
    options: RenderOptions,
) -> bool {
    // Span boundaries are character offsets in the source, not Markdown
    // token boundaries. Rendering the three pieces independently can change
    // the meaning of inline markers or fenced blocks, so preserve the
    // existing fragment/card rendering for Markdown-derived content.
    if is_markdown_content(&target.extractor) {
        return false;
    }
    let Some((merged, center_start, center_end)) = merge_dig_window(target, window) else {
        return false;
    };
    let Some(center_seq) = target.seq else {
        return false;
    };

    render_dig_compact_metadata(target, window, options);

    let prefix = char_slice(&merged, 0, center_start);
    let center = char_slice(&merged, center_start, center_end);
    let suffix = char_slice(&merged, center_end, merged.chars().count());
    if !prefix.is_empty() {
        println!();
        render::print_line(options, Intent::Muted, "Before");
        print_compact_part(&prefix, Intent::Muted, options);
    }
    println!();
    render::print_line(
        options,
        Intent::Info,
        &format!("Target · chunk {center_seq}"),
    );
    print_compact_part(&center, Intent::Info, options);
    if !suffix.is_empty() {
        println!();
        render::print_line(options, Intent::Muted, "After");
        print_compact_part(&suffix, Intent::Muted, options);
    }
    println!();
    render::print_line(options, Intent::Muted, short_id(&target.digest));
    true
}

/// Merge adjacent chunk text using character spans, removing only verified
/// overlap. Gaps or inconsistent metadata deliberately force fallback.
fn merge_dig_window(
    target: &ChunkWithRefs,
    window: &[ChunkWithRefs],
) -> Option<(String, usize, usize)> {
    let target_span = target.span?;
    let first_span = window.first()?.span?;
    let mut merged = String::new();
    let range_start = first_span.start;
    let mut range_end = first_span.start;

    for chunk in window {
        let span = chunk.span?;
        let text_len = chunk.text.chars().count();
        if span.segment != target_span.segment
            || span.start > span.end
            || span.end - span.start != u64::try_from(text_len).ok()?
            || span.start < range_start
            || span.start > range_end
        {
            return None;
        }

        let overlap = range_end.saturating_sub(span.start);
        let skip = usize::try_from(overlap).ok()?;
        if skip > text_len {
            return None;
        }
        if skip > 0 {
            let merged_len = merged.chars().count();
            let existing = char_slice(&merged, merged_len.checked_sub(skip)?, merged_len);
            let incoming = char_slice(&chunk.text, 0, skip);
            if existing != incoming {
                return None;
            }
        }
        merged.extend(chunk.text.chars().skip(skip));
        range_end = range_end.max(span.end);
    }

    let center_start = usize::try_from(target_span.start.checked_sub(range_start)?).ok()?;
    let center_end = usize::try_from(target_span.end.checked_sub(range_start)?).ok()?;
    if center_end > merged.chars().count() || center_start > center_end {
        return None;
    }
    Some((merged, center_start, center_end))
}

fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

fn print_compact_part(text: &str, intent: Intent, options: RenderOptions) {
    if text.is_empty() {
        return;
    }
    let rendered = markdown_render::render_fragment(text, false, options);
    for line in rendered.lines() {
        render::print_line(options, intent, &format!("  {line}"));
    }
    if rendered.ends_with('\n') {
        println!();
    }
}

/// DOCX extraction is intentionally canonical Markdown, so it shares the
/// terminal renderer with native Markdown. Other extractors produce plain
/// text even when their content happens to contain Markdown-like characters.
fn is_markdown_content(extractor: &str) -> bool {
    matches!(extractor, "markdown" | "docx")
}

/// Preserve the card indentation on every rendered line.
fn print_indented(indent: &str, text: &str) {
    for line in text.lines() {
        println!("{indent}{line}");
    }
    if text.ends_with('\n') {
        println!();
    }
}

fn render_dig_extended(
    _input: &str,
    target: &ChunkWithRefs,
    window: &[ChunkWithRefs],
    radius: u32,
) {
    let options = RenderOptions::stdout();
    render_dig_header(&target.digest, radius, options);
    render_dig_cards_body(target, window, options);
    println!();
    println!("Window: center_seq={:?}, radius={radius}", target.seq);
    for chunk in window {
        println!("  {}", short_id(&chunk.digest));
        for reference in &chunk.refs {
            println!("    {} ({})", reference.path.as_str(), reference.status);
        }
    }
}

fn chunk_stale(chunk: &ChunkWithRefs) -> bool {
    chunk
        .refs
        .iter()
        .any(|reference| reference.status == lode_core::index::records::FileStatus::Stale)
}

/// Escape one field in the stable table format. A table row is always one
/// physical line, while this representation keeps every field reversible.
fn escape_tsv(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\t' => escaped.push_str("\\t"),
            '\r' => escaped.push_str("\\r"),
            '\n' => escaped.push_str("\\n"),
            character if character.is_control() => {
                escaped.push_str(&format!(r"\u{{{:x}}}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

/// Write one table record. Table output deliberately has no terminal-width
/// logic: stdout can be piped and every field remains complete.
fn print_tsv(fields: &[&str]) {
    let row = fields
        .iter()
        .map(|field| escape_tsv(field))
        .collect::<Vec<_>>()
        .join("\t");
    println!("{row}");
}

fn render_dig_table(window: &[ChunkWithRefs], center_digest: &str, center_seq: Option<u32>) {
    print_tsv(&[
        "SEQ", "CENTER", "STATE", "DIGEST", "PATH", "HEADING", "PAGE", "TEXT",
    ]);
    for chunk in window {
        let seq = chunk
            .seq
            .map_or_else(String::new, |value| value.to_string());
        let center =
            if chunk.digest == center_digest && (center_seq.is_none() || chunk.seq == center_seq) {
                "yes"
            } else {
                "no"
            };
        let state = if chunk_stale(chunk) { "stale" } else { "fresh" };
        let path = chunk.primary().path.as_str().to_string();
        let heading = chunk.heading.clone();
        let page = chunk.page.map_or_else(String::new, |page| page.to_string());
        let fields = [
            seq.as_str(),
            center,
            state,
            chunk.digest.as_str(),
            path.as_str(),
            heading.as_str(),
            page.as_str(),
            chunk.text.as_str(),
        ];
        print_tsv(&fields);
    }
}

///
/// Runs a silent detection first so the stale bits are fresh before search
/// reads them — this command writes `files.status` (it is not read-only).
/// It needs an existing index: with none, it short-circuits with the
/// `no_index` message.
fn prospect(
    workspace: &std::path::Path,
    query: String,
    top_k: Option<u32>,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("prospect", workspace, view) {
        return code;
    }
    if query.trim().is_empty() {
        let text = lode_core::messages::require("invalid_query");
        return ui_msg::die(
            view,
            "prospect",
            error_code::INVALID_QUERY,
            text.error,
            text.hint,
        );
    }

    let settings = match configure_output(workspace, "prospect", view, palette, no_color) {
        Ok(settings) => settings,
        Err(code) => return code,
    };
    let embedder = match build_embedder(&settings.embedding) {
        Ok(e) => e,
        Err(e) => {
            return ui_msg::die(
                view,
                "prospect",
                error_code::EMBEDDING,
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
        return ui_msg::die(view, "prospect", error_code::NO_INDEX, &error, text.hint);
    }
    let mut store = match Store::open_existing_with_tokenizer(&db_path, &settings.fts.strategy) {
        Ok(s) => s,
        Err(e) => {
            return ui_msg::die(
                view,
                "prospect",
                error_code::INDEX_OPEN,
                &format!("Cannot open the lode index: {e}"),
                Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
            );
        }
    };

    // Refresh the stale bits before searching so per-chunk annotation and
    // the library-wide dirty signal reflect the current workspace.
    let ignore_files = settings
        .app
        .ignore
        .sources
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if let Err(e) = detect_changes(&mut store, workspace, &ignore_files) {
        return ui_msg::die(
            view,
            "prospect",
            error_code::SCAN,
            &format!("Could not finish scanning the workspace: {e}"),
            Some("Fix the underlying error and rerun `lode prospect`."),
        );
    }

    let top_k = top_k.unwrap_or(settings.retrieval.top_k);
    if top_k == 0 {
        return ui_msg::die(
            view,
            "prospect",
            error_code::INVALID_TOP_K,
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
                view,
                "prospect",
                error_code::SEARCH,
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
fn survey(
    workspace: &std::path::Path,
    view: View,
    palette: Option<PaletteArg>,
    no_color: bool,
) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("survey", workspace, view) {
        return code;
    }
    let settings = match configure_output(workspace, "survey", view, palette, no_color) {
        Ok(settings) => settings,
        Err(code) => return code,
    };
    let ignore_files = settings
        .app
        .ignore
        .sources
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    let db_path = workspace.join(INDEX_DB_RELATIVE);
    let has_index = db_path.is_file();
    log::debug!("index database: {}", db_path.display());

    let result = if has_index {
        match Store::open_existing(&db_path) {
            Ok(mut store) => match detect_changes(&mut store, workspace, &ignore_files) {
                Ok(result) => {
                    log::info!("detected {} pending changes", result.pending());
                    result
                }
                Err(e) => {
                    return ui_msg::die(
                        view,
                        "survey",
                        error_code::SCAN,
                        &format!("Could not finish scanning the workspace: {e}"),
                        Some("Fix the underlying error and rerun `lode survey`."),
                    );
                }
            },
            Err(e) => {
                return ui_msg::die(
                    view,
                    "survey",
                    error_code::INDEX_OPEN,
                    &format!("Cannot open the lode index: {e}"),
                    Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
                );
            }
        }
    } else {
        // No index yet: classify against an empty snapshot (all new).
        log::info!("no index found; classifying against an empty snapshot");
        let indexed: HashMap<WorkspacePath, FileRecord> = HashMap::new();
        classify(&indexed, workspace, &ignore_files)
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
fn mine(
    workspace: &std::path::Path,
    view: View,
    from_scratch: bool,
    palette: Option<PaletteArg>,
    no_color: bool,
    no_progress: bool,
) -> u8 {
    if let Some(code) = ui_msg::bad_workspace("mine", workspace, view) {
        return code;
    }

    let settings = match configure_output(workspace, "mine", view, palette, no_color) {
        Ok(settings) => settings,
        Err(code) => return code,
    };
    let ignore_files = settings
        .app
        .ignore
        .sources
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let embedder = match build_embedder(&settings.embedding) {
        Ok(e) => e,
        Err(e) => {
            return ui_msg::die(
                view,
                "mine",
                error_code::EMBEDDING,
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
                    view,
                    "mine",
                    error_code::CHUNKING,
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
                view,
                "mine",
                error_code::DIMENSION,
                &format!("Could not determine the embedding dimension: {e}"),
                Some(
                    "Check your embedding endpoint, or set `embedding.model_dimension` in lode.toml.",
                ),
            );
        }
        if let Err(e) = embedder.model_id() {
            return ui_msg::die(
                view,
                "mine",
                error_code::MODEL,
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
                    view,
                    "mine",
                    error_code::RESET,
                    &format!("Could not reset the lode index: {e}"),
                    Some("Ensure the workspace is writable."),
                );
            }
        }
    }

    let result = if has_index {
        match Store::open_existing_with_tokenizer(&db_path, &tokenizer) {
            Ok(mut store) => match detect_changes(&mut store, workspace, &ignore_files) {
                Ok(detect) => {
                    let bar = mine_progress_bar(view, no_progress);
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
                            if let Some(bar) = &bar {
                                bar.finish_and_clear();
                            }
                            return ui_msg::die(
                                view,
                                "mine",
                                error_code::MINING,
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
                        view,
                        "mine",
                        error_code::SCAN,
                        &format!("Could not finish scanning the workspace: {e}"),
                        Some("Fix the underlying error and rerun `lode mine`."),
                    );
                }
            },
            Err(e) => {
                return ui_msg::die(
                    view,
                    "mine",
                    error_code::INDEX_OPEN,
                    &format!("Cannot open the lode index: {e}"),
                    Some("Ensure `.lode/index.db` is intact, or delete it and remine."),
                );
            }
        }
    } else {
        // No index yet: classify against an empty snapshot.
        log::info!("no index found; classifying against an empty snapshot");
        let indexed: HashMap<WorkspacePath, FileRecord> = HashMap::new();
        let detect = classify(&indexed, workspace, &ignore_files);
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
                        view,
                        "mine",
                        error_code::DIMENSION,
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
                        view,
                        "mine",
                        error_code::MODEL,
                        &format!("Could not determine the embedding model: {e}"),
                        Some(
                            "Check your embedding endpoint, or set `embedding.model` in lode.toml.",
                        ),
                    );
                }
            };
            match Store::open(&db_path, &model_id, dimension, &tokenizer) {
                Ok(mut store) => {
                    let bar = mine_progress_bar(view, no_progress);
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
                            if let Some(bar) = &bar {
                                bar.finish_and_clear();
                            }
                            return ui_msg::die(
                                view,
                                "mine",
                                error_code::MINING,
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
                        view,
                        "mine",
                        error_code::INDEX_CREATE,
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
    if result.failed.is_empty() {
        EXIT_OK
    } else {
        EXIT_PARTIAL_FAILURE
    }
}

/// Build the mine progress bar: an indicatif spinner bar on stderr.
///
/// Suppressed for `--view json` (JSON consumers want a clean stream) and
/// hidden automatically when stderr is not a terminal (indicatif detects
/// this), so `lode mine > log` stays unpolluted.
fn mine_progress_bar(view: View, no_progress: bool) -> Option<ProgressBar> {
    if !should_show_progress(
        view,
        no_progress,
        std::io::stderr().is_terminal(),
        std::env::var("TERM").ok().as_deref(),
    ) {
        return None;
    }
    let options = RenderOptions::stderr();
    let template = progress_template(options.palette(), options.color());
    let style = ProgressStyle::with_template(&template)
        .expect("progress template is valid")
        .progress_chars("##-");
    let bar = ProgressBar::new_spinner().with_style(style);
    bar.set_draw_target(ProgressDrawTarget::stderr());
    Some(bar)
}

fn progress_template(palette: OutputPalette, color: bool) -> String {
    if !color {
        return "{spinner} mining {pos}/{len} {msg}".to_owned();
    }
    // Indicatif delegates styled placeholders to console, which applies its
    // own process-global NO_COLOR check. Embed the already-resolved colour
    // here so an explicit `no_color = false` is honoured consistently.
    format!(
        "{}{{spinner}}\x1b[0m mining {{pos}}/{{len}} {{msg}}",
        progress_color(palette)
    )
}

fn progress_color(palette: OutputPalette) -> &'static str {
    match palette {
        OutputPalette::Ansi => "\x1b[32m",
        OutputPalette::AccessibleTol => "\x1b[38;2;0;153;136m",
        OutputPalette::AccessibleOkabeIto => "\x1b[38;2;0;158;115m",
    }
}

fn should_show_progress(
    view: View,
    no_progress: bool,
    stderr_terminal: bool,
    term: Option<&str>,
) -> bool {
    !no_progress
        && !matches!(view, View::Json | View::Table)
        && stderr_terminal
        && !term.is_some_and(|term| term.eq_ignore_ascii_case("dumb"))
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
        render::print_line(RenderOptions::stdout(), Intent::Info, "Nothing to do.");
        return;
    }
    let failed = !result.failed.is_empty();
    let options = RenderOptions::stdout();
    render::print_line(
        options,
        if failed {
            Intent::Warning
        } else {
            Intent::Success
        },
        mine_header(result),
    );
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

/// Render the mine result as stable TSV: one complete row per outcome.
fn render_mine_table(_workspace: &std::path::Path, result: &SyncSummary) {
    print_tsv(&["STATUS", "PATH", "FROM", "DETAIL"]);
    for path in &result.added {
        print_tsv(&["added", path.as_str(), "", ""]);
    }
    for path in &result.updated {
        print_tsv(&["updated", path.as_str(), "", ""]);
    }
    for (from, to) in &result.renamed {
        print_tsv(&["renamed", to.as_str(), from.as_str(), ""]);
    }
    for path in &result.removed {
        print_tsv(&["removed", path.as_str(), "", ""]);
    }
    for failure in &result.failed {
        print_tsv(&["failed", failure.path.as_str(), "", failure.error.as_str()]);
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
    let options = RenderOptions::stdout();
    let preview_w = terminal_preview_width();
    for (index, hit) in hits.iter().enumerate() {
        let rank = index + 1;
        let width = hits.len().to_string().len();
        println!();
        render::print_line(
            options,
            Intent::Info,
            &format!("{rank:>width$}  {}", hit_source_line(hit), width = width),
        );
        println!("     {}", truncate_tail(&preview(&hit.text), preview_w));
        render::print_line(
            options,
            Intent::Muted,
            &format!("     {:.3} · {}", hit.score, short_id(&hit.digest)),
        );
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

/// Render prospect hits as stable TSV records.
fn render_prospect_table(hits: &[lode_core::index::search::SearchHit]) {
    print_tsv(&[
        "RANK", "SCORE", "DIGEST", "PATH", "HEADING", "PAGE", "STATE",
    ]);
    for (index, hit) in hits.iter().enumerate() {
        let rank = index + 1;
        let primary = hit.primary();
        let state = if hit.stale() { "stale" } else { "fresh" };
        let rank = rank.to_string();
        let score = hit.score.to_string();
        let path = primary.path.as_str().to_string();
        let heading = hit.heading.clone();
        let page = hit.page.map_or_else(String::new, |page| page.to_string());
        let fields = [
            rank.as_str(),
            score.as_str(),
            hit.digest.as_str(),
            path.as_str(),
            heading.as_str(),
            page.as_str(),
            state,
        ];
        print_tsv(&fields);
    }
    if hits.iter().any(|hit| hit.stale()) {
        print_prospect_stale_warning();
    }
}

fn print_prospect_stale_warning() {
    eprintln!(
        "Warning: results include stale files; verify them before relying on them. Run `lode mine` to update."
    );
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
        render::print_line(
            RenderOptions::stdout(),
            Intent::Warning,
            "Warning: results include stale files; verify them before relying on them. Run `lode mine` to update.",
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

/// Render the survey as stable TSV: one complete row per pending change.
fn render_survey_table(result: &DetectResult, has_index: bool) {
    let _ = has_index;
    print_tsv(&["STATUS", "PATH", "FROM", "DETAIL"]);
    for change in &result.changes {
        match change {
            Change::Added(snap) => {
                print_tsv(&["new", snap.path.as_str(), "", ""]);
            }
            Change::Modified { old, new } => {
                let detail = format!(
                    "{} -> {} bytes, {} -> {}",
                    old.size,
                    new.size,
                    format_unix_ts(old.mtime),
                    format_unix_ts(new.mtime),
                );
                print_tsv(&["modified", new.path.as_str(), "", detail.as_str()]);
            }
            Change::Removed(record) => print_tsv(&["missing", record.path.as_str(), "", ""]),
            Change::Renamed { from, to } => {
                print_tsv(&["renamed", to.as_str(), from.as_str(), ""]);
            }
        }
    }
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

#[cfg(test)]
mod dig_tests {
    use super::*;

    #[test]
    fn normalize_digest_accepts_prefixed_bare_and_case_insensitive_hex() {
        assert_eq!(
            normalize_digest(" blake3:AbC123 ").as_deref(),
            Some("abc123")
        );
        assert_eq!(normalize_digest("DeAd").as_deref(), Some("dead"));
    }

    #[test]
    fn normalize_digest_rejects_empty_and_non_hex() {
        assert!(normalize_digest("").is_none());
        assert!(normalize_digest("blake3:").is_none());
        assert!(normalize_digest("#dead").is_none());
        assert!(normalize_digest("dead-gold").is_none());
    }

    #[test]
    fn dig_chunk_json_contains_text_only_for_full_chunks() {
        let chunk = ChunkWithRefs {
            digest: "blake3:abc".into(),
            extractor: "markdown".into(),
            text: "ore".into(),
            heading: "A".into(),
            seq: Some(2),
            page: None,
            span: None,
            refs: vec![],
        };
        assert!(dig_chunk_json(&chunk).get("text").is_some());
        assert!(dig_chunk_json_without_text(&chunk).get("text").is_none());
    }

    #[test]
    fn merge_dig_window_removes_verified_overlap() {
        use lode_core::index::records::ChunkWithRefs;
        use lode_core::ingestion::types::ChunkSpan;

        let make_chunk = |text: &str, seq: u32, start: u64, end: u64| ChunkWithRefs {
            digest: format!("blake3:{seq}"),
            extractor: "text".into(),
            text: text.into(),
            heading: "A".into(),
            seq: Some(seq),
            page: None,
            span: Some(ChunkSpan {
                segment: 0,
                start,
                end,
            }),
            refs: vec![],
        };
        let before = make_chunk("abcdef", 0, 0, 6);
        let target = make_chunk("efghij", 1, 4, 10);
        let after = make_chunk("ij", 2, 8, 10);
        let window = vec![before, target.clone(), after];

        let (merged, center_start, center_end) = merge_dig_window(&target, &window).unwrap();
        assert_eq!(merged, "abcdefghij");
        assert_eq!(&char_slice(&merged, center_start, center_end), "efghij");

        let repeated_before = make_chunk("aaa", 0, 0, 3);
        let repeated_target = make_chunk("aaa", 1, 2, 5);
        let (repeated, _, _) = merge_dig_window(
            &repeated_target,
            &[repeated_before, repeated_target.clone()],
        )
        .unwrap();
        assert_eq!(repeated, "aaaaa");
    }

    #[test]
    fn merge_dig_window_falls_back_on_gaps_or_missing_spans() {
        use lode_core::index::records::ChunkWithRefs;
        use lode_core::ingestion::types::ChunkSpan;

        let chunk = ChunkWithRefs {
            digest: "blake3:abc".into(),
            extractor: "text".into(),
            text: "abc".into(),
            heading: "A".into(),
            seq: Some(0),
            page: None,
            span: Some(ChunkSpan {
                segment: 0,
                start: 0,
                end: 3,
            }),
            refs: vec![],
        };
        let missing = ChunkWithRefs {
            span: None,
            ..chunk.clone()
        };
        assert!(merge_dig_window(&chunk, &[missing]).is_none());

        let gap = ChunkWithRefs {
            digest: "blake3:def".into(),
            text: "def".into(),
            seq: Some(1),
            span: Some(ChunkSpan {
                segment: 0,
                start: 5,
                end: 8,
            }),
            ..chunk.clone()
        };
        let window = vec![chunk.clone(), gap];
        assert!(merge_dig_window(&chunk, &window).is_none());

        let wrong_length = ChunkWithRefs {
            text: "abcd".into(),
            span: Some(ChunkSpan {
                segment: 0,
                start: 0,
                end: 3,
            }),
            ..chunk.clone()
        };
        let window = vec![wrong_length.clone()];
        assert!(merge_dig_window(&wrong_length, &window).is_none());

        let other_segment = ChunkWithRefs {
            span: Some(ChunkSpan {
                segment: 1,
                start: 3,
                end: 6,
            }),
            ..chunk.clone()
        };
        let window = vec![chunk.clone(), other_segment];
        assert!(merge_dig_window(&chunk, &window).is_none());

        let conflicting_overlap = ChunkWithRefs {
            text: "ZZdef".into(),
            span: Some(ChunkSpan {
                segment: 0,
                start: 1,
                end: 6,
            }),
            ..chunk.clone()
        };
        let window = vec![chunk.clone(), conflicting_overlap];
        assert!(merge_dig_window(&chunk, &window).is_none());
    }

    #[test]
    fn markdown_compact_falls_back_to_cards() {
        use lode_core::index::records::ChunkWithRefs;
        use lode_core::ingestion::types::ChunkSpan;

        let chunk = ChunkWithRefs {
            digest: "blake3:markdown".into(),
            extractor: "markdown".into(),
            text: "**bold**".into(),
            heading: "A".into(),
            seq: Some(0),
            page: None,
            span: Some(ChunkSpan {
                segment: 0,
                start: 0,
                end: 8,
            }),
            refs: vec![],
        };
        assert!(!render_dig_compact_body(
            &chunk,
            std::slice::from_ref(&chunk),
            RenderOptions::plain()
        ));
    }
}

#[cfg(test)]
mod output_contract_tests {
    use super::*;

    #[test]
    fn tsv_escaping_preserves_record_boundaries() {
        assert_eq!(escape_tsv("a\\b\tc\rd\ne\u{0000}"), r"a\\b\tc\rd\ne\u{0}");
    }

    #[test]
    fn tsv_line_has_one_physical_line_and_stable_columns() {
        let line = ["status", "path", "", "detail\nwith tab\t"]
            .iter()
            .map(|field| escape_tsv(field))
            .collect::<Vec<_>>()
            .join("\t");
        assert_eq!(line, "status\tpath\t\tdetail\\nwith tab\\t");
        assert_eq!(line.lines().count(), 1);
        assert_eq!(line.split('\t').count(), 4);
    }

    #[test]
    fn progress_policy_is_limited_to_interactive_human_views() {
        assert!(should_show_progress(
            View::Compact,
            false,
            true,
            Some("xterm-256color")
        ));
        assert!(should_show_progress(View::Extended, false, true, None));
        assert!(!should_show_progress(
            View::Compact,
            true,
            true,
            Some("xterm")
        ));
        assert!(!should_show_progress(
            View::Compact,
            false,
            false,
            Some("xterm")
        ));
        assert!(!should_show_progress(
            View::Compact,
            false,
            true,
            Some("dumb")
        ));
        assert!(!should_show_progress(
            View::Table,
            false,
            true,
            Some("xterm")
        ));
        assert!(!should_show_progress(
            View::Json,
            false,
            true,
            Some("xterm")
        ));
    }

    #[test]
    fn progress_template_uses_resolved_palette_and_color_switch() {
        let palette = OutputPalette::AccessibleOkabeIto;
        assert_eq!(
            progress_template(palette, true),
            "\x1b[38;2;0;158;115m{spinner}\x1b[0m mining {pos}/{len} {msg}"
        );

        assert_eq!(
            progress_template(palette, false),
            "{spinner} mining {pos}/{len} {msg}"
        );
    }
}
