//! Shared presentation primitives for human-readable CLI output.
//!
//! Domain data stays in lode-core. This module owns the presentation axes:
//! View selects the layout, while Intent describes the meaning of a rendered
//! line. Keeping those axes separate lets the layout evolve without making
//! storage or search aware of terminal styling.

use std::io::IsTerminal;

use termimad::crossterm::style::{Attribute, Color, ContentStyle};

/// Presentation meaning of a human-readable output fragment.
///
/// This is intentionally a CLI concept. A core status such as Stale is a
/// domain fact; the renderer may present it with Warning, while the same
/// fact could be represented differently by JSON or another frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Intent {
    /// A successful or positive outcome.
    Success,
    /// A blocking failure or invalid operation.
    Error,
    /// Neutral guidance or section information.
    Info,
    /// A condition that deserves attention but does not block output.
    Warning,
    /// Supporting metadata that should not compete with the content.
    Muted,
}

/// Terminal presentation settings shared by human-readable renderers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RenderOptions {
    ansi: bool,
}

impl RenderOptions {
    /// Resolve stdout capabilities for one human-readable render.
    pub(crate) fn stdout() -> Self {
        Self {
            ansi: std::io::stdout().is_terminal(),
        }
    }

    /// Resolve stderr capabilities for one human-readable diagnostic.
    pub(crate) fn stderr() -> Self {
        Self {
            ansi: std::io::stderr().is_terminal(),
        }
    }

    /// Build a deterministic no-colour context for tests and piped output.
    #[cfg(test)]
    pub(crate) const fn plain() -> Self {
        Self { ansi: false }
    }

    pub(crate) const fn ansi(self) -> bool {
        self.ansi
    }

    /// Paint one human-readable fragment according to its presentation intent.
    pub(crate) fn paint(self, intent: Intent, text: &str) -> String {
        if !self.ansi {
            return text.to_owned();
        }
        intent_style(intent).apply(text).to_string()
    }
}

/// Print one line while keeping styling out of command-specific formatting.
pub(crate) fn print_line(options: RenderOptions, intent: Intent, text: &str) {
    println!("{}", options.paint(intent, text));
}

fn intent_style(intent: Intent) -> ContentStyle {
    let mut style = ContentStyle::new();
    style.foreground_color = Some(match intent {
        Intent::Success => Color::Green,
        Intent::Error => Color::Red,
        Intent::Info => Color::Cyan,
        Intent::Warning => Color::Yellow,
        Intent::Muted => Color::DarkGrey,
    });
    if matches!(intent, Intent::Error | Intent::Warning) {
        style.attributes.set(Attribute::Bold);
    }
    style
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_output_does_not_add_ansi() {
        let options = RenderOptions::plain();
        assert_eq!(options.paint(Intent::Warning, "attention"), "attention");
    }

    #[test]
    fn intents_are_distinct_from_layout() {
        assert_ne!(Intent::Info, Intent::Muted);
        assert_ne!(Intent::Success, Intent::Error);
    }
}
