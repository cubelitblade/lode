//! Terminal rendering for Markdown fragments.
//!
//! Chunks are not guaranteed to be complete Markdown documents: the splitter
//! may cut through a block or an inline construct. Termimad gives these
//! fragments a consistent terminal skin and wrapping policy, while keeping
//! the stored Markdown untouched. It does not invent missing context at a
//! chunk boundary; that remains an ingestion concern if the conservative
//! fallback proves insufficient.
#![warn(clippy::pedantic)]

use termimad::MadSkin;
use termimad::crossterm::style::{Attribute, Color};

use crate::render::RenderOptions;
use lode_core::config::OutputPalette;

/// Render one stored Markdown fragment for a human-facing terminal view.
pub(crate) fn render_fragment(text: &str, markdown: bool, options: RenderOptions) -> String {
    if !markdown {
        return text.to_owned();
    }

    let skin = terminal_skin(options);
    skin.term_text(text).to_string()
}

fn terminal_skin(options: RenderOptions) -> MadSkin {
    if !options.ansi() {
        return MadSkin::no_style();
    }

    let mut skin = MadSkin::default();
    let (info, muted, success, warning) = match options.palette() {
        OutputPalette::Ansi => (Color::Cyan, Color::DarkGrey, Color::Green, Color::Yellow),
        OutputPalette::AccessibleTol => (
            Color::Rgb {
                r: 0,
                g: 119,
                b: 187,
            },
            Color::Rgb {
                r: 128,
                g: 128,
                b: 128,
            },
            Color::Rgb {
                r: 0,
                g: 153,
                b: 136,
            },
            Color::Rgb {
                r: 238,
                g: 119,
                b: 51,
            },
        ),
        OutputPalette::AccessibleOkabeIto => (
            Color::Rgb {
                r: 86,
                g: 180,
                b: 233,
            },
            Color::Rgb {
                r: 136,
                g: 136,
                b: 136,
            },
            Color::Rgb {
                r: 0,
                g: 158,
                b: 115,
            },
            Color::Rgb {
                r: 240,
                g: 228,
                b: 66,
            },
        ),
    };
    skin.headers[0].set_fg(info);
    skin.headers[0].add_attr(Attribute::Bold);
    skin.headers[1].set_fg(info);
    skin.headers[1].add_attr(Attribute::Bold);
    skin.bold.set_fg(warning);
    skin.inline_code.set_fg(success);
    skin.quote_mark.set_fg(muted);
    skin.horizontal_rule.set_fg(muted);
    skin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_common_markdown_without_source_markers() {
        let rendered = render_fragment(
            "# Title\n\n**bold** and `code`",
            true,
            RenderOptions::plain(),
        );
        assert!(rendered.contains("Title"));
        assert!(rendered.contains("bold"));
        assert!(rendered.contains("code"));
        assert!(!rendered.contains("**bold**"));
        assert!(!rendered.contains("`code`"));
    }

    #[test]
    fn renders_lists_and_quotes() {
        let rendered = render_fragment("> quote\n\n- one\n- two", true, RenderOptions::plain());
        assert!(rendered.contains("quote"));
        assert!(rendered.contains("one"));
        assert!(rendered.contains("two"));
    }

    #[test]
    fn does_not_guess_a_leading_code_fence() {
        let rendered = render_fragment("code body\n\n```", true, RenderOptions::plain());
        assert!(rendered.contains("code body"));
        assert!(!rendered.contains("```"));
    }

    #[test]
    fn non_markdown_is_byte_preserving() {
        let source = "# plain\n\n**text**";
        assert_eq!(
            render_fragment(source, false, RenderOptions::plain()),
            source
        );
    }
}
