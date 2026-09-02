use clap::{Parser, Subcommand};

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
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report ingestion / index status (alias: status).
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
    std::process::ExitCode::from(dispatch(cli.command))
}

/// Route a parsed subcommand to its implementation.
///
/// Each branch is a placeholder until the command lands. Using `eprintln` +
/// non-zero exit keeps misuse loud instead of silently succeeding.
fn dispatch(command: Command) -> u8 {
    match command {
        Command::Survey => todo_command("survey"),
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
