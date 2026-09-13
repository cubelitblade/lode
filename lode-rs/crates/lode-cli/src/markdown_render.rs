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

/// Render one stored Markdown fragment for a human-facing terminal view.
pub(crate) fn render_fragment(text: &str, markdown: bool, options: RenderOptions) -> String {
    if !markdown {
        return text.to_owned();
    }

    let skin = terminal_skin(options.ansi());
    skin.term_text(text).to_string()
}

fn terminal_skin(ansi: bool) -> MadSkin {
    if !ansi {
        return MadSkin::no_style();
    }

    let mut skin = MadSkin::default();
    skin.headers[0].set_fg(Color::Cyan);
    skin.headers[0].add_attr(Attribute::Bold);
    skin.headers[1].set_fg(Color::Blue);
    skin.headers[1].add_attr(Attribute::Bold);
    skin.bold.set_fg(Color::Yellow);
    skin.inline_code.set_fg(Color::Green);
    skin.quote_mark.set_fg(Color::DarkGrey);
    skin.horizontal_rule.set_fg(Color::DarkGrey);
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
