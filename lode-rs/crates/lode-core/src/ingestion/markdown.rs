//! Markdown-aware segmentation for canonical extracted documents.
//!
//! The splitter deliberately keeps the source Markdown intact. It only uses
//! parser offsets to establish heading boundaries and plain-text provenance;
//! it does not serialize events back into Markdown.
#![warn(clippy::pedantic)]

use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

use crate::ingestion::types::{HEADING_SEP, Segment};

#[derive(Debug)]
struct Heading {
    start: usize,
    level: usize,
    text: String,
}

/// Split canonical Markdown into heading-aware segments.
pub(crate) fn into_segments(markdown: &str) -> Vec<Segment> {
    if markdown.trim().is_empty() {
        return Vec::new();
    }

    let headings = collect_headings(markdown);
    if headings.is_empty() {
        return vec![Segment {
            text: markdown.to_owned(),
            heading: String::new(),
            page: None,
        }];
    }

    let mut segments = Vec::new();
    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    let mut cursor = 0;

    for (index, heading) in headings.iter().enumerate() {
        if heading.start > cursor {
            push_segment(
                &mut segments,
                &markdown[cursor..heading.start],
                &heading_stack,
            );
        }

        heading_stack.retain(|(level, _)| *level < heading.level);
        heading_stack.push((heading.level, heading.text.clone()));
        cursor = heading.start;

        if index + 1 == headings.len() {
            push_segment(&mut segments, &markdown[cursor..], &heading_stack);
        }
    }

    segments
}

fn push_segment(segments: &mut Vec<Segment>, text: &str, heading_stack: &[(usize, String)]) {
    if text.trim().is_empty() {
        return;
    }
    let heading = heading_stack
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>()
        .join(HEADING_SEP);
    segments.push(Segment {
        text: text.to_owned(),
        heading,
        page: None,
    });
}

fn collect_headings(markdown: &str) -> Vec<Heading> {
    let mut headings = Vec::new();
    let mut current: Option<(usize, usize, String)> = None;

    for (event, range) in Parser::new(markdown).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                current = Some((range.start, heading_level(level), String::new()));
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some((start, level, text)) = current.take() {
                    headings.push(Heading { start, level, text });
                }
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some((_, _, heading)) = current.as_mut() {
                    heading.push_str(&text);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some((_, _, heading)) = current.as_mut() {
                    heading.push(' ');
                }
            }
            _ => {}
        }
    }
    headings
}

fn heading_level(level: HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_markdown_and_builds_heading_chain() {
        let segments = into_segments(
            "preamble\n\n# Report\n\n**intro**\n\n## Details\n\nbody\n\n# Appendix\n\nend",
        );
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].heading, "");
        assert_eq!(segments[0].text, "preamble\n\n");
        assert_eq!(segments[1].heading, "Report");
        assert_eq!(segments[1].text, "# Report\n\n**intro**\n\n");
        assert_eq!(segments[2].heading, "Report / Details");
        assert_eq!(segments[2].text, "## Details\n\nbody\n\n");
        assert_eq!(segments[3].heading, "Appendix");
        assert_eq!(segments[3].text, "# Appendix\n\nend");
    }

    #[test]
    fn recognizes_setext_headings() {
        let segments = into_segments("Title\n=====\n\nbody\n\nNext\n----\n\nend");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].heading, "Title");
        assert_eq!(segments[0].text, "Title\n=====\n\nbody\n\n");
        assert_eq!(segments[1].heading, "Title / Next");
    }

    #[test]
    fn ignores_heading_like_text_inside_fenced_code() {
        let segments = into_segments("# Real\n\n```text\n# not a heading\n```\n\nbody");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].heading, "Real");
        assert!(segments[0].text.contains("# not a heading"));
    }

    #[test]
    fn uses_visible_text_for_formatted_heading() {
        let segments = into_segments("# **Bold** [title](https://example.test)\n\nbody");
        assert_eq!(segments[0].heading, "Bold title");
    }

    #[test]
    fn empty_input_has_no_segments() {
        assert!(into_segments("\n  \n").is_empty());
    }
}
