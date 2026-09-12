//! Text extraction: file bytes + suffix -> structured segments.
//!
//! Unsupported files return `None`; supported formats that cannot be parsed
//! return an [`ExtractionError`] so the sync pipeline can report one failed
//! file without discarding the rest of the run.
#![warn(clippy::pedantic)]

use std::io::Cursor;

use office_oxide::docx::{BlockElement, DocxDocument, Paragraph, ParagraphContent, RunContent};
use pdf_oxide::converters::{ConversionOptions, ReadingOrderMode};
use pdf_oxide::outline::{Destination, OutlineItem};

use crate::ingestion::formats::PLAIN_EXTENSIONS;
use crate::ingestion::types::{HEADING_SEP, Segment};

/// Errors raised while parsing a supported document format.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// The DOCX package or its `WordprocessingML` contents were malformed.
    #[error("could not parse DOCX document: {0}")]
    Docx(String),
    /// The PDF structure or page content could not be parsed.
    #[error("could not parse PDF document: {0}")]
    Pdf(String),
    /// The PDF is encrypted and has not been authenticated.
    #[error("encrypted PDF requires authentication")]
    EncryptedPdf,
}

/// Structured extraction for a supported file, or `None` for an unsupported
/// extension.
///
/// # Errors
///
/// Returns [`ExtractionError`] when a supported document cannot be parsed.
pub fn extract_document(
    data: &[u8],
    suffix: &str,
) -> Result<Option<Vec<Segment>>, ExtractionError> {
    let suffix = suffix.to_ascii_lowercase();
    if PLAIN_EXTENSIONS.contains(&suffix.as_str()) {
        return Ok(Some(vec![Segment {
            text: decode_text(data),
            heading: String::new(),
            page: None,
        }]));
    }
    match suffix.as_str() {
        ".docx" => extract_docx(data).map(Some),
        ".pdf" => extract_pdf(data).map(Some),
        _ => Ok(None),
    }
}

/// Decode plain text with the same permissive fallback used by the Python
/// reference implementation.
#[must_use]
pub fn decode_text(data: &[u8]) -> String {
    if let Some(body) = data.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        if let Ok(text) = String::from_utf8(body.to_vec()) {
            return text;
        }
    } else if let Ok(text) = String::from_utf8(data.to_vec()) {
        return text;
    }

    if data.starts_with(&[0xff, 0xfe]) && data.len().is_multiple_of(2) {
        let units = data[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        if let Ok(text) = String::from_utf16(&units) {
            return text;
        }
    }
    if data.starts_with(&[0xfe, 0xff]) && data.len().is_multiple_of(2) {
        let units = data[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_be_bytes(*bytes))
            .collect::<Vec<_>>();
        if let Ok(text) = String::from_utf16(&units) {
            return text;
        }
    }

    data.iter().map(|&byte| char::from(byte)).collect()
}

fn extract_docx(data: &[u8]) -> Result<Vec<Segment>, ExtractionError> {
    let document = DocxDocument::from_reader(Cursor::new(data.to_vec()))
        .map_err(|error| ExtractionError::Docx(error.to_string()))?;
    let mut output = SegmentAccumulator::default();

    for block in &document.body.elements {
        match block {
            BlockElement::Paragraph(paragraph) => {
                let style_id = paragraph
                    .properties
                    .as_ref()
                    .and_then(|properties| properties.style_id.as_deref());
                let style_name = style_id
                    .and_then(|id| document.styles.as_ref()?.styles.get(id)?.name.as_deref());
                output.push_paragraph(
                    paragraph_text(paragraph),
                    heading_level(style_id, style_name),
                );
            }
            BlockElement::Table(table) => {
                for row in &table.rows {
                    let cells = row
                        .cells
                        .iter()
                        .map(|cell| blocks_text(&cell.content))
                        .collect::<Vec<_>>();
                    output.push_body(cells.join(" | "));
                }
            }
        }
    }
    Ok(output.finish())
}

fn paragraph_text(paragraph: &Paragraph) -> String {
    let mut output = String::new();
    let mut textbox_separator_pending = false;
    for content in &paragraph.content {
        let runs = match content {
            ParagraphContent::Run(run) => std::slice::from_ref(run),
            ParagraphContent::Hyperlink(link) => link.runs.as_slice(),
        };
        for run in runs {
            for item in &run.content {
                match item {
                    RunContent::Text(text) => {
                        if !text.is_empty() {
                            if textbox_separator_pending {
                                output.push('\n');
                                textbox_separator_pending = false;
                            }
                            output.push_str(text);
                        }
                    }
                    RunContent::Break(_) => {
                        textbox_separator_pending = false;
                        output.push('\n');
                    }
                    RunContent::Tab => {
                        // A tab is inline content; keep it adjacent to a
                        // preceding textbox and let the next text run consume
                        // the lazy textbox boundary.
                        output.push('\t');
                    }
                    RunContent::TextBox(blocks) => {
                        let boxed = blocks_text(blocks);
                        if !boxed.is_empty() {
                            if textbox_separator_pending {
                                output.push('\n');
                            }
                            if !output.is_empty() && !output.ends_with('\n') {
                                output.push('\n');
                            }
                            output.push_str(&boxed);
                            textbox_separator_pending = true;
                        }
                    }
                    RunContent::Drawing(_) => {}
                }
            }
        }
    }
    output
}

fn blocks_text(blocks: &[BlockElement]) -> String {
    let mut output = Vec::new();
    for block in blocks {
        match block {
            BlockElement::Paragraph(paragraph) => {
                let text = paragraph_text(paragraph);
                if !text.trim().is_empty() {
                    output.push(text);
                }
            }
            BlockElement::Table(table) => {
                for row in &table.rows {
                    output.push(
                        row.cells
                            .iter()
                            .map(|cell| blocks_text(&cell.content))
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                }
            }
        }
    }
    output.join("\n")
}

fn heading_level(style_id: Option<&str>, style_name: Option<&str>) -> Option<usize> {
    style_id
        .and_then(heading_level_from_label)
        .or_else(|| style_name.and_then(heading_level_from_label))
}

fn heading_level_from_label(label: &str) -> Option<usize> {
    let normalized = label.trim().to_ascii_lowercase();
    if normalized == "title" {
        return Some(0);
    }
    let level = normalized.strip_prefix("heading")?.trim().parse().ok()?;
    (1..=9).contains(&level).then_some(level)
}

#[derive(Default)]
struct SegmentAccumulator {
    segments: Vec<Segment>,
    heading_stack: Vec<(usize, String)>,
    current_heading: String,
    current_text: Vec<String>,
}

impl SegmentAccumulator {
    fn push_paragraph(&mut self, text: String, level: Option<usize>) {
        if text.trim().is_empty() {
            return;
        }
        if let Some(level) = level {
            self.flush();
            if level == 0 {
                self.heading_stack = vec![(level, text.trim().to_owned())];
            } else {
                self.heading_stack
                    .retain(|(heading_level, _)| *heading_level < level);
                self.heading_stack.push((level, text.trim().to_owned()));
            }
            self.current_heading = self
                .heading_stack
                .iter()
                .map(|(_, heading)| heading.as_str())
                .collect::<Vec<_>>()
                .join(HEADING_SEP);
        }
        self.current_text.push(text);
    }

    fn push_body(&mut self, text: String) {
        if !text.trim().is_empty() {
            self.current_text.push(text);
        }
    }

    fn flush(&mut self) {
        if self.current_text.is_empty() {
            return;
        }
        self.segments.push(Segment {
            text: self.current_text.join("\n"),
            heading: self.current_heading.clone(),
            page: None,
        });
        self.current_text.clear();
    }

    fn finish(mut self) -> Vec<Segment> {
        self.flush();
        self.segments
    }
}

fn extract_pdf(data: &[u8]) -> Result<Vec<Segment>, ExtractionError> {
    let document = pdf_oxide::PdfDocument::from_bytes(data.to_vec())
        .map_err(|error| ExtractionError::Pdf(error.to_string()))?;
    if document.is_encrypted() && !document.is_authenticated() {
        return Err(ExtractionError::EncryptedPdf);
    }

    let options = ConversionOptions {
        extract_tables: false,
        reading_order_mode: ReadingOrderMode::TopToBottomLeftToRight,
        ..Default::default()
    };
    let heading_events = collect_outline(&document)?;
    let page_count = document
        .page_count()
        .map_err(|error| ExtractionError::Pdf(error.to_string()))?;
    let mut heading_stack = Vec::new();
    let mut event_index = 0;
    let mut segments = Vec::new();

    for page_index in 0..page_count {
        while event_index < heading_events.len() && heading_events[event_index].0 <= page_index {
            let (_, level, title) = &heading_events[event_index];
            heading_stack.truncate(*level);
            heading_stack.push(title.clone());
            event_index += 1;
        }

        let has_widget_annotations = page_has_widget_annotations(&document, page_index)?;
        let assembled = document
            .extract_text_with_options(page_index, &options)
            .map_err(|error| ExtractionError::Pdf(error.to_string()))?;
        let text = if contains_rtl(&assembled) || has_widget_annotations {
            assembled
        } else {
            let lines = document
                .extract_text_lines(page_index)
                .map_err(|error| ExtractionError::Pdf(error.to_string()))?;
            lines
                .into_iter()
                .map(|line| line.text)
                .collect::<Vec<_>>()
                .join("\n")
        };
        let text = text.trim().to_owned();
        if text.is_empty() {
            continue;
        }
        segments.push(Segment {
            text,
            heading: heading_stack.join(HEADING_SEP),
            page: Some(
                u32::try_from(page_index + 1)
                    .map_err(|error| ExtractionError::Pdf(error.to_string()))?,
            ),
        });
    }
    Ok(segments)
}

fn collect_outline(
    document: &pdf_oxide::PdfDocument,
) -> Result<Vec<(usize, usize, String)>, ExtractionError> {
    let Some(outline) = document
        .get_outline()
        .map_err(|error| ExtractionError::Pdf(error.to_string()))?
    else {
        return Ok(Vec::new());
    };
    let mut events = Vec::new();
    collect_outline_items(&outline, 0, &mut events);
    events.sort_by_key(|(page, _, _)| *page);
    Ok(events)
}

fn collect_outline_items(
    items: &[OutlineItem],
    level: usize,
    events: &mut Vec<(usize, usize, String)>,
) {
    for item in items {
        match &item.dest {
            Some(Destination::PageIndex(page)) => {
                events.push((*page, level, item.title.clone()));
            }
            Some(Destination::Named(name)) => {
                log::warn!("ignoring unresolved PDF named destination {name:?}");
            }
            None => {
                log::warn!(
                    "ignoring PDF outline item without destination: {:?}",
                    item.title
                );
            }
        }
        let child_level = if matches!(item.dest, Some(Destination::PageIndex(_))) {
            level + 1
        } else {
            // An unresolved parent cannot contribute a title to the active
            // stack; do not let it create a phantom nesting level for valid
            // descendants.
            level
        };
        collect_outline_items(&item.children, child_level, events);
    }
}

fn contains_rtl(text: &str) -> bool {
    text.chars().any(|character| {
        matches!(
            character as u32,
            0x0590..=0x08ff | 0xfb1d..=0xfdff | 0xfe70..=0xfeff
        )
    })
}

fn page_has_widget_annotations(
    document: &pdf_oxide::PdfDocument,
    page_index: usize,
) -> Result<bool, ExtractionError> {
    let page = document
        .get_page(page_index)
        .map_err(|error| ExtractionError::Pdf(error.to_string()))?;
    let Some(dictionary) = page.as_dict() else {
        return Ok(false);
    };
    let Some(annotations) = dictionary.get("Annots") else {
        return Ok(false);
    };
    let annotations = match annotations {
        pdf_oxide::object::Object::Array(items) => items.clone(),
        pdf_oxide::object::Object::Reference(reference) => match document.load_object(*reference) {
            Ok(pdf_oxide::object::Object::Array(items)) => items,
            Ok(_) => return Ok(false),
            Err(error) => return Err(ExtractionError::Pdf(error.to_string())),
        },
        _ => return Ok(false),
    };
    for annotation in annotations {
        let object = match annotation {
            pdf_oxide::object::Object::Reference(reference) => document
                .load_object(reference)
                .map_err(|error| ExtractionError::Pdf(error.to_string()))?,
            object => object,
        };
        let Some(dictionary) = object.as_dict() else {
            continue;
        };
        if dictionary
            .get("Subtype")
            .and_then(pdf_oxide::object::Object::as_name)
            .is_some_and(|subtype| subtype.eq_ignore_ascii_case("Widget"))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_utf8_and_bom() {
        assert_eq!(decode_text("héllo".as_bytes()), "héllo");
        assert_eq!(decode_text(&[0xef, 0xbb, 0xbf, b'h', b'i']), "hi");
    }

    #[test]
    fn decode_utf16_and_latin1() {
        let mut data = vec![0xff, 0xfe];
        for unit in "héllo".encode_utf16() {
            data.extend_from_slice(&unit.to_le_bytes());
        }
        assert_eq!(decode_text(&data), "héllo");
        assert_eq!(decode_text(&[0x63, 0x61, 0x66, 0xe9]), "café");
    }

    #[test]
    fn unsupported_formats_are_none() {
        assert!(extract_document(b"x", ".png").unwrap().is_none());
    }

    #[test]
    fn heading_labels_are_conservative() {
        assert_eq!(heading_level_from_label("Title"), Some(0));
        assert_eq!(heading_level_from_label("Heading 3"), Some(3));
        assert_eq!(heading_level_from_label("Custom"), None);
    }

    #[test]
    fn heading_stack_replaces_same_level_without_title() {
        let mut output = SegmentAccumulator::default();
        output.push_paragraph("First".to_owned(), Some(1));
        output.push_body("first body".to_owned());
        output.push_paragraph("Second".to_owned(), Some(1));
        output.push_body("second body".to_owned());
        output.push_paragraph("Nested".to_owned(), Some(2));
        output.push_body("nested body".to_owned());
        output.push_paragraph("Third".to_owned(), Some(1));
        output.push_body("third body".to_owned());

        let segments = output.finish();
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.heading.as_str())
                .collect::<Vec<_>>(),
            ["First", "Second", "Second / Nested", "Third",]
        );
    }

    #[test]
    fn paragraph_text_preserves_runs_breaks_tabs_and_textbox_boundaries() {
        let paragraph = Paragraph {
            properties: None,
            content: vec![
                ParagraphContent::Run(office_oxide::docx::Run {
                    properties: None,
                    content: vec![RunContent::Text("Before".to_owned())],
                }),
                ParagraphContent::Hyperlink(office_oxide::docx::Hyperlink {
                    target: office_oxide::docx::HyperlinkTarget::External(
                        "https://example.test".to_owned(),
                    ),
                    tooltip: None,
                    runs: vec![office_oxide::docx::Run {
                        properties: None,
                        content: vec![RunContent::Text("Link".to_owned())],
                    }],
                }),
                ParagraphContent::Run(office_oxide::docx::Run {
                    properties: None,
                    content: vec![
                        RunContent::TextBox(vec![BlockElement::Paragraph(Paragraph {
                            properties: None,
                            content: vec![ParagraphContent::Run(office_oxide::docx::Run {
                                properties: None,
                                content: vec![RunContent::Text("Box".to_owned())],
                            })],
                        })]),
                        RunContent::Tab,
                        RunContent::Break(office_oxide::docx::BreakType::Line),
                        RunContent::Text("After".to_owned()),
                    ],
                }),
            ],
        };

        assert_eq!(paragraph_text(&paragraph), "BeforeLink\nBox\t\nAfter");

        let breaks = Paragraph {
            properties: None,
            content: vec![ParagraphContent::Run(office_oxide::docx::Run {
                properties: None,
                content: vec![
                    RunContent::Text("top".to_owned()),
                    RunContent::Break(office_oxide::docx::BreakType::Line),
                    RunContent::Break(office_oxide::docx::BreakType::Line),
                    RunContent::Text("bottom".to_owned()),
                ],
            })],
        };
        assert_eq!(paragraph_text(&breaks), "top\n\nbottom");
    }

    #[test]
    fn nested_table_blocks_keep_row_and_block_order() {
        let paragraph = |text: &str| {
            BlockElement::Paragraph(Paragraph {
                properties: None,
                content: vec![ParagraphContent::Run(office_oxide::docx::Run {
                    properties: None,
                    content: vec![RunContent::Text(text.to_owned())],
                })],
            })
        };
        let nested = BlockElement::Table(office_oxide::docx::Table {
            properties: None,
            grid: Vec::new(),
            rows: vec![office_oxide::docx::TableRow {
                properties: None,
                cells: vec![office_oxide::docx::TableCell {
                    properties: None,
                    content: vec![paragraph("inner")],
                }],
            }],
        });
        let outer = office_oxide::docx::Table {
            properties: None,
            grid: Vec::new(),
            rows: vec![office_oxide::docx::TableRow {
                properties: None,
                cells: vec![office_oxide::docx::TableCell {
                    properties: None,
                    content: vec![paragraph("before"), nested, paragraph("after")],
                }],
            }],
        };

        assert_eq!(
            blocks_text(&[BlockElement::Table(outer)]),
            "before\ninner\nafter"
        );
    }

    #[test]
    fn extracts_docx_sections_and_tables() {
        let mut writer = office_oxide::docx::write::DocxWriter::new();
        writer
            .add_heading("Report", 1)
            .add_paragraph("Overview")
            .add_heading("Details", 2)
            .add_paragraph("Body")
            .add_table(&[vec!["Name", "Value"], vec!["A", "1"]]);
        let mut bytes = Cursor::new(Vec::new());
        writer.write_to(&mut bytes).unwrap();

        let segments = extract_document(bytes.get_ref(), ".docx").unwrap().unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].heading, "Report");
        assert_eq!(segments[0].text, "Report\nOverview");
        assert_eq!(segments[1].heading, "Report / Details");
        assert_eq!(segments[1].text, "Details\nBody\nName | Value\nA | 1");
    }

    #[test]
    fn extracts_pdf_pages_and_rejects_corrupt_input() {
        let mut pdf = pdf_oxide::api::Pdf::from_text("first page").unwrap();
        let bytes = pdf.save_to_bytes().unwrap();
        let segments = extract_document(&bytes, ".pdf").unwrap().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].page, Some(1));
        assert!(segments[0].text.contains("first page"));

        let error = extract_document(b"not a pdf", ".pdf").unwrap_err();
        assert!(matches!(error, ExtractionError::Pdf(_)));
    }

    #[test]
    fn unresolved_outline_parent_does_not_create_phantom_depth() {
        let items = vec![OutlineItem {
            title: "Unresolved".to_owned(),
            dest: None,
            children: vec![OutlineItem {
                title: "Child".to_owned(),
                dest: Some(Destination::PageIndex(2)),
                children: Vec::new(),
            }],
        }];
        let mut events = Vec::new();
        collect_outline_items(&items, 0, &mut events);
        assert_eq!(events, vec![(2, 0, "Child".to_owned())]);
    }
}
