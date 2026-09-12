//! Text extraction: file bytes + suffix -> structured segments.
//!
//! Unsupported files return `None`; supported formats that cannot be parsed
//! return an [`ExtractionError`] so the sync pipeline can report one failed
//! file without discarding the rest of the run.
#![warn(clippy::pedantic)]

use std::io::Cursor;

use office_oxide::docx::{BlockElement, ParagraphContent, RunContent};
use office_oxide::ir_render::{ImageEmbed, MarkdownOptions};
use office_oxide::{Document, DocumentFormat};
use pdf_oxide::converters::{ConversionOptions, ReadingOrderMode};
use pdf_oxide::outline::{Destination, OutlineItem};

use crate::ingestion::formats::{MARKDOWN_EXTENSIONS, TEXT_EXTENSIONS};
use crate::ingestion::markdown::into_segments;
use crate::ingestion::types::{HEADING_SEP, Segment};

/// Errors raised while parsing a supported document format.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// The legacy Word binary document could not be parsed.
    #[error("could not parse legacy DOC document: {0}")]
    Doc(String),
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
    if TEXT_EXTENSIONS.contains(&suffix.as_str()) {
        return Ok(Some(vec![Segment {
            text: decode_text(data),
            heading: String::new(),
            page: None,
        }]));
    }
    if MARKDOWN_EXTENSIONS.contains(&suffix.as_str()) {
        return Ok(Some(into_segments(&decode_text(data))));
    }
    match suffix.as_str() {
        ".doc" => extract_doc(data).map(Some),
        ".docx" => extract_docx(data).map(Some),
        ".pdf" => extract_pdf(data).map(Some),
        _ => Ok(None),
    }
}

fn extract_doc(data: &[u8]) -> Result<Vec<Segment>, ExtractionError> {
    let text = rwml::extract_text(data).map_err(|error| ExtractionError::Doc(error.to_string()))?;
    let text = text.trim().to_owned();
    if text.is_empty() {
        return Err(ExtractionError::Doc(
            "document contains no indexable text".to_owned(),
        ));
    }
    Ok(vec![Segment {
        text,
        heading: String::new(),
        page: None,
    }])
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
    let document = Document::from_reader(Cursor::new(data.to_vec()), DocumentFormat::Docx)
        .map_err(|error| ExtractionError::Docx(error.to_string()))?;
    let title = document.as_docx().and_then(find_docx_title);
    let mut ir = document.to_ir();
    promote_docx_title(&mut ir, title.as_deref());
    for section in &mut ir.sections {
        // Preserve the existing body-only indexing contract. Headers and
        // footers are repeated display furniture, not document body content.
        section.header = None;
        section.footer = None;
        section.first_page_header = None;
        section.first_page_footer = None;
        section.even_page_header = None;
        section.even_page_footer = None;
        section.elements.retain(|element| {
            !matches!(
                element,
                office_oxide::ir::Element::Footnote(_) | office_oxide::ir::Element::Endnote(_)
            )
        });
    }
    let markdown = ir.to_markdown_with(MarkdownOptions {
        image_embed: ImageEmbed::None,
    });
    Ok(into_segments(&markdown))
}

/// Find the first non-empty paragraph using Word's built-in `Title` style.
///
/// Word's `Title` is document-level metadata in the visual hierarchy, not a
/// sibling of `Heading 1`. We therefore detect it from the source DOCX style
/// before the format-agnostic converter turns the body into IR.
fn find_docx_title(document: &office_oxide::docx::DocxDocument) -> Option<String> {
    let styles = document.styles.as_ref()?;
    document.body.elements.iter().find_map(|element| {
        let BlockElement::Paragraph(paragraph) = element else {
            return None;
        };
        let style_id = paragraph
            .properties
            .as_ref()
            .and_then(|properties| properties.style_id.as_deref())?;
        if !is_title_style(style_id, styles) {
            return None;
        }
        let text = docx_paragraph_text(paragraph);
        (!text.trim().is_empty()).then(|| text.trim().to_owned())
    })
}

fn is_title_style(style_id: &str, styles: &office_oxide::docx::StyleSheet) -> bool {
    let mut current = Some(style_id);
    for _ in 0..20 {
        let Some(id) = current else {
            break;
        };
        if id.eq_ignore_ascii_case("title") {
            return true;
        }
        let Some(style) = styles.styles.get(id) else {
            break;
        };
        if style
            .name
            .as_deref()
            .is_some_and(|name| name.trim().eq_ignore_ascii_case("title"))
        {
            return true;
        }
        current = style.based_on.as_deref();
    }
    false
}

fn docx_paragraph_text(paragraph: &office_oxide::docx::Paragraph) -> String {
    paragraph
        .content
        .iter()
        .flat_map(|content| match content {
            ParagraphContent::Run(run) => run_content_text(&run.content),
            ParagraphContent::Hyperlink(link) => link
                .runs
                .iter()
                .flat_map(|run| run_content_text(&run.content))
                .collect::<Vec<_>>(),
        })
        .collect()
}

fn run_content_text(content: &[RunContent]) -> Vec<String> {
    content
        .iter()
        .flat_map(|item| match item {
            RunContent::Text(text) => vec![text.clone()],
            RunContent::Break(_) => vec!["\n".to_owned()],
            RunContent::Tab => vec!["\t".to_owned()],
            RunContent::Drawing(_) => Vec::new(),
            RunContent::TextBox(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    BlockElement::Paragraph(paragraph) => Some(docx_paragraph_text(paragraph)),
                    BlockElement::Table(_) => None,
                })
                .collect(),
        })
        .collect()
}

/// Normalize DOCX title semantics before rendering Markdown.
///
/// A source `Title` becomes Markdown H1. Existing Word heading levels are
/// shifted down one level only when that title exists; documents without a
/// `Title` retain their ordinary `Heading 1` → H1 mapping. The converter also
/// synthesizes a section title from the first heading, which would otherwise
/// duplicate content in the Markdown, so section titles are cleared here.
fn promote_docx_title(ir: &mut office_oxide::DocumentIR, title: Option<&str>) {
    for section in &mut ir.sections {
        section.title = None;
    }
    let Some(title) = title else {
        return;
    };

    let title_position = ir
        .sections
        .iter()
        .enumerate()
        .find_map(|(section_index, section)| {
            section
                .elements
                .iter()
                .enumerate()
                .find_map(|(element_index, element)| {
                    let text = match element {
                        office_oxide::ir::Element::Paragraph(paragraph) => {
                            ir_inline_text(&paragraph.content)
                        }
                        office_oxide::ir::Element::Heading(heading) => {
                            ir_inline_text(&heading.content)
                        }
                        _ => return None,
                    };
                    (text.trim() == title.trim()).then_some((section_index, element_index))
                })
        });
    let Some((title_section, title_element)) = title_position else {
        return;
    };

    if let office_oxide::ir::Element::Paragraph(paragraph) =
        &mut ir.sections[title_section].elements[title_element]
    {
        let paragraph = std::mem::take(paragraph);
        ir.sections[title_section].elements[title_element] =
            office_oxide::ir::Element::Heading(office_oxide::ir::Heading {
                level: 1,
                content: paragraph.content,
                frame_position: paragraph.frame_position,
                alignment: paragraph.alignment,
            });
    } else if let office_oxide::ir::Element::Heading(heading) =
        &mut ir.sections[title_section].elements[title_element]
    {
        heading.level = 1;
    }

    for (section_index, section) in ir.sections.iter_mut().enumerate() {
        for (element_index, element) in section.elements.iter_mut().enumerate() {
            if (section_index, element_index) == (title_section, title_element) {
                continue;
            }
            if let office_oxide::ir::Element::Heading(heading) = element {
                heading.level = heading.level.saturating_add(1).min(6);
            }
        }
    }
}

fn ir_inline_text(content: &[office_oxide::ir::InlineContent]) -> String {
    content
        .iter()
        .filter_map(|item| match item {
            office_oxide::ir::InlineContent::Text(span) => Some(span.text.as_str()),
            office_oxide::ir::InlineContent::LineBreak => Some("\n"),
            _ => None,
        })
        .collect()
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
    fn extracts_markdown_with_raw_source_and_heading() {
        let source = "\n# **标题**\r\n\r\n正文";
        let segments = extract_document(source.as_bytes(), ".md").unwrap().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].heading, "标题");
        assert_eq!(segments[0].text, source);
        assert!(segments[0].page.is_none());

        let mut utf8_bom = vec![0xef, 0xbb, 0xbf];
        utf8_bom.extend_from_slice(source.as_bytes());
        let segments = extract_document(&utf8_bom, ".md").unwrap().unwrap();
        assert_eq!(segments[0].text, source);
        assert_eq!(segments[0].heading, "标题");

        let mut utf16 = vec![0xff, 0xfe];
        for unit in source.encode_utf16() {
            utf16.extend_from_slice(&unit.to_le_bytes());
        }
        let segments = extract_document(&utf16, ".markdown").unwrap().unwrap();
        assert_eq!(segments[0].text, source);
        assert!(segments[0].page.is_none());
    }

    #[test]
    fn extracts_txt_as_one_unstructured_segment() {
        let source = "# This stays plain text\n\nbody";
        let segments = extract_document(source.as_bytes(), ".txt")
            .unwrap()
            .unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, source);
        assert!(segments[0].heading.is_empty());
        assert!(segments[0].page.is_none());
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
        assert_eq!(segments[0].text, "# **Report**\n\nOverview\n\n");
        assert_eq!(segments[1].heading, "Report / Details");
        assert_eq!(
            segments[1].text,
            "## **Details**\n\nBody\n\n| Name | Value |\n| --- | --- |\n| A | 1 |"
        );
    }

    #[test]
    fn rejects_corrupt_legacy_doc_input() {
        let error = extract_document(b"not an OLE2 compound file", ".doc").unwrap_err();
        assert!(matches!(error, ExtractionError::Doc(_)));
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
