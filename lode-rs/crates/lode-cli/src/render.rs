//! Shared presentation primitives for human-readable CLI output.
//!
//! Domain data stays in lode-core. This module owns the presentation axes:
//! View selects the layout, while Intent describes the meaning of a rendered
//! line. Keeping those axes separate lets the layout evolve without making
//! storage or search aware of terminal styling.

use std::io::IsTerminal;
use std::sync::{Mutex, OnceLock};

use lode_core::config::OutputPalette;
use termimad::crossterm::style::{Attribute, Color, Colored, ContentStyle};

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
    style: bool,
    color: bool,
    palette: OutputPalette,
}

#[derive(Clone, Copy)]
struct OutputState {
    palette: OutputPalette,
    no_color: Option<bool>,
}

static OUTPUT_STATE: OnceLock<Mutex<OutputState>> = OnceLock::new();

/// Set the effective output configuration for this CLI invocation.
pub(crate) fn set_output(palette: OutputPalette, no_color: Option<bool>) {
    let state = OUTPUT_STATE.get_or_init(|| {
        Mutex::new(OutputState {
            palette: OutputPalette::Ansi,
            no_color: None,
        })
    });
    *state.lock().expect("output state is not poisoned") = OutputState { palette, no_color };
    // Crossterm also consults NO_COLOR globally while formatting styles. Keep
    // that switch aligned with the explicit configuration so
    // `no_color = false` can intentionally override NO_COLOR, matching the
    // Python/Rich contract. Terminal capability checks still happen per
    // stream in `RenderOptions::for_stream`.
    let disabled = no_color
        .unwrap_or_else(|| std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()));
    Colored::set_ansi_color_disabled(disabled);
}

fn output_state() -> OutputState {
    *OUTPUT_STATE
        .get_or_init(|| {
            Mutex::new(OutputState {
                palette: OutputPalette::Ansi,
                no_color: None,
            })
        })
        .lock()
        .expect("output state is not poisoned")
}

impl RenderOptions {
    fn from_capabilities(
        state: OutputState,
        terminal: bool,
        term_dumb: bool,
        env_no_color: bool,
    ) -> Self {
        let no_color = state.no_color.unwrap_or(env_no_color);
        Self {
            style: terminal && !term_dumb,
            color: terminal && !no_color && !term_dumb,
            palette: state.palette,
        }
    }

    fn for_stream(terminal: bool) -> Self {
        let state = output_state();
        let env_no_color = std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
        let dumb = std::env::var("TERM").is_ok_and(|term| term.eq_ignore_ascii_case("dumb"));
        Self::from_capabilities(state, terminal, dumb, env_no_color)
    }

    /// Resolve stdout capabilities for one human-readable render.
    pub(crate) fn stdout() -> Self {
        Self::for_stream(std::io::stdout().is_terminal())
    }

    /// Resolve stderr capabilities for one human-readable diagnostic.
    pub(crate) fn stderr() -> Self {
        Self::for_stream(std::io::stderr().is_terminal())
    }

    /// Build a deterministic no-colour context for tests and piped output.
    #[cfg(test)]
    pub(crate) const fn plain() -> Self {
        Self {
            style: false,
            color: false,
            palette: OutputPalette::Ansi,
        }
    }

    pub(crate) const fn ansi(self) -> bool {
        self.style
    }

    pub(crate) const fn color(self) -> bool {
        self.color
    }

    pub(crate) const fn palette(self) -> OutputPalette {
        self.palette
    }

    /// Paint one human-readable fragment according to its presentation intent.
    pub(crate) fn paint(self, intent: Intent, text: &str) -> String {
        if !self.style {
            return text.to_owned();
        }
        let mut style = intent_style(self.palette, intent);
        if !self.color {
            style.foreground_color = None;
        }
        style.apply(text).to_string()
    }
}

/// Print one line while keeping styling out of command-specific formatting.
pub(crate) fn print_line(options: RenderOptions, intent: Intent, text: &str) {
    crate::write_stdout(format_args!("{}", options.paint(intent, text)));
}

fn intent_style(palette: OutputPalette, intent: Intent) -> ContentStyle {
    let mut style = ContentStyle::new();
    style.foreground_color = Some(match palette {
        OutputPalette::Ansi => match intent {
            Intent::Success => Color::Green,
            Intent::Error => Color::Red,
            Intent::Info => Color::Cyan,
            Intent::Warning => Color::Yellow,
            Intent::Muted => Color::DarkGrey,
        },
        OutputPalette::AccessibleTol => match intent {
            Intent::Success => Color::Rgb {
                r: 0,
                g: 153,
                b: 136,
            },
            Intent::Error => Color::Rgb {
                r: 204,
                g: 51,
                b: 17,
            },
            Intent::Info => Color::Rgb {
                r: 0,
                g: 119,
                b: 187,
            },
            Intent::Warning => Color::Rgb {
                r: 238,
                g: 119,
                b: 51,
            },
            Intent::Muted => Color::Rgb {
                r: 128,
                g: 128,
                b: 128,
            },
        },
        OutputPalette::AccessibleOkabeIto => match intent {
            Intent::Success => Color::Rgb {
                r: 0,
                g: 158,
                b: 115,
            },
            Intent::Error => Color::Rgb {
                r: 213,
                g: 94,
                b: 0,
            },
            Intent::Info => Color::Rgb {
                r: 86,
                g: 180,
                b: 233,
            },
            Intent::Warning => Color::Rgb {
                r: 240,
                g: 228,
                b: 66,
            },
            Intent::Muted => Color::Rgb {
                r: 136,
                g: 136,
                b: 136,
            },
        },
    });
    if matches!(intent, Intent::Error | Intent::Warning) {
        style.attributes.set(Attribute::Bold);
    }
    style
}

#[cfg(test)]
mod tests {
    use super::*;

    static OUTPUT_TEST_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn accessible_palette_uses_fixed_rgb_colour() {
        let _guard = OUTPUT_TEST_LOCK.lock().expect("test lock is not poisoned");
        set_output(OutputPalette::AccessibleTol, Some(false));
        let rendered = intent_style(OutputPalette::AccessibleTol, Intent::Success)
            .apply("ok")
            .to_string();
        assert!(rendered.contains("38;2;0;153;136"), "rendered={rendered:?}");
        set_output(OutputPalette::Ansi, None);
    }

    #[test]
    fn explicit_no_color_disables_foreground_color() {
        let _guard = OUTPUT_TEST_LOCK.lock().expect("test lock is not poisoned");
        let options = RenderOptions {
            style: true,
            color: false,
            palette: OutputPalette::AccessibleOkabeIto,
        };
        let rendered = options.paint(Intent::Error, "failed");
        assert!(rendered.contains("failed"));
        assert!(rendered.contains("\x1b[1m"));
        assert!(!rendered.contains("38;"));
    }

    #[test]
    fn terminal_policy_handles_no_color_and_dumb_terminal() {
        let state = OutputState {
            palette: OutputPalette::Ansi,
            no_color: None,
        };
        assert!(RenderOptions::from_capabilities(state, true, false, false).color);
        assert!(!RenderOptions::from_capabilities(state, true, false, true).color);
        assert!(!RenderOptions::from_capabilities(state, false, false, false).style);
        assert!(!RenderOptions::from_capabilities(state, true, true, false).style);
        let forced_on = OutputState {
            no_color: Some(false),
            ..state
        };
        assert!(RenderOptions::from_capabilities(forced_on, true, false, true).color);
    }
}
