//! Markdown-aware segmentation for canonical extracted documents.
//!
//! The splitter deliberately keeps the source Markdown intact. It only uses
//! parser offsets to establish heading boundaries and plain-text provenance;
//! it does not serialize events back into Markdown.
#![warn(clippy::pedantic)]

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

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
            let prefix = &markdown[cursor..heading.start];
            // Pure separators stay attached to the heading they introduce so
            // reconstruction preserves the decoded source byte-for-byte.
            if !prefix.trim().is_empty() {
                push_segment(&mut segments, prefix, &heading_stack);
                cursor = heading.start;
            }
        }

        heading_stack.retain(|(level, text)| *level < heading.level && !text.is_empty());
        if !heading.text.is_empty() {
            heading_stack.push((heading.level, heading.text.clone()));
        }

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
    let mut nested_depth = 0usize;

    let options = Options::ENABLE_YAML_STYLE_METADATA_BLOCKS | Options::ENABLE_FOOTNOTES;
    for (event, range) in Parser::new_ext(markdown, options).into_offset_iter() {
        match event {
            Event::Start(
                Tag::BlockQuote(_) | Tag::List(_) | Tag::Item | Tag::FootnoteDefinition(_),
            ) => nested_depth += 1,
            Event::End(
                TagEnd::BlockQuote(_) | TagEnd::List(_) | TagEnd::Item | TagEnd::FootnoteDefinition,
            ) => nested_depth = nested_depth.saturating_sub(1),
            Event::Start(Tag::Heading { level, .. }) => {
                if nested_depth == 0 {
                    current = Some((range.start, heading_level(level), String::new()));
                }
            }
            Event::End(TagEnd::Heading(_)) => {
                if nested_depth == 0
                    && let Some((start, level, text)) = current.take()
                {
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

    #[test]
    fn preserves_whitespace_before_first_heading() {
        let source = "\n\n# Title\n\nbody";
        let segments = into_segments(source);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, source);
        assert_eq!(segments[0].heading, "Title");
    }

    #[test]
    fn ignores_nested_container_headings() {
        let source = "# Top\n\n> # Quoted\n\n- # Listed\n\nbody";
        let segments = into_segments(source);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].heading, "Top");
        assert_eq!(segments[0].text, source);
    }

    #[test]
    fn preserves_yaml_and_ignores_footnote_heading() {
        let source =
            "---\ntitle: Example\n---\n\n# Top\n\n[^note]:\n    # Footnote heading\n\nbody";
        let segments = into_segments(source);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].heading, "");
        assert_eq!(segments[0].text, "---\ntitle: Example\n---\n\n");
        assert_eq!(segments[1].heading, "Top");
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<String>(),
            source
        );
    }

    #[test]
    fn empty_heading_does_not_create_empty_provenance() {
        let segments = into_segments("#\n\n## Child\n\nbody");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].heading, "");
        assert_eq!(segments[1].heading, "Child");
    }

    #[test]
    fn quality_cases_preserve_source_and_provenance() {
        let cases = [
            ("# A\n\n### C\n\n## B\n\ntext", vec!["A", "A / C", "A / B"]),
            (
                "Title\n===\n\nNext\n---\n\ntext",
                vec!["Title", "Title / Next"],
            ),
            ("plain text without headings", vec![""]),
            (
                "```md\n# code\n```\n\n\\# escaped\n\n    # indented",
                vec![""],
            ),
            ("# One\r\n\r\n# Two\r\n", vec!["One", "Two"]),
        ];
        for (source, expected) in cases {
            let segments = into_segments(source);
            let reconstructed: String = segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect();
            assert_eq!(reconstructed, source);
            let headings: Vec<&str> = segments
                .iter()
                .map(|segment| segment.heading.as_str())
                .collect();
            assert_eq!(headings, expected);
        }
    }
}
