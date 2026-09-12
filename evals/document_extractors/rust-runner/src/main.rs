use std::env;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use serde::Serialize;

#[derive(Serialize)]
struct Segment {
    text: String,
    heading: String,
    page: Option<u32>,
}

#[derive(Serialize)]
struct Output {
    candidate: String,
    format: String,
    status: &'static str,
    text: String,
    markdown: Option<String>,
    segments: Vec<Segment>,
    warnings: Vec<String>,
    elapsed_ns: u128,
    error: Option<String>,
}

struct CandidateExtraction {
    segments: Vec<Segment>,
    markdown: Option<String>,
    warnings: Vec<String>,
}

impl CandidateExtraction {
    fn text_only(segments: Vec<Segment>) -> Self {
        Self {
            segments,
            markdown: None,
            warnings: Vec::new(),
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: document-extractor-rust-runner CANDIDATE FORMAT INPUT");
        return ExitCode::from(2);
    }
    let candidate = &args[1];
    let format = &args[2];
    let path = Path::new(&args[3]);
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read {}: {error}", path.display());
            return ExitCode::from(2);
        }
    };
    let started = Instant::now();
    let result = match (format.as_str(), candidate.as_str()) {
        ("docx", "lode_core") | ("pdf", "lode_core") => {
            extract_lode_core(&bytes, format).map(CandidateExtraction::text_only)
        }
        ("docx", "office_oxide") => {
            extract_office_oxide(&bytes).map(CandidateExtraction::text_only)
        }
        ("docx", "rwml") => extract_rwml(&bytes).map(CandidateExtraction::text_only),
        ("doc", "office_oxide") => extract_office_oxide_doc(&bytes),
        ("doc", "rwml") => extract_rwml_doc(&bytes),
        ("docx", "docx_rs") => extract_docx_rs(&bytes).map(CandidateExtraction::text_only),
        ("docx", "rs_docx") => extract_rs_docx(&bytes).map(CandidateExtraction::text_only),
        ("pdf", "pdf_oxide") => extract_pdf_oxide(&bytes),
        ("pdf", "pdf_oxide_remediated") => extract_pdf_oxide_remediated(&bytes),
        ("pdf", "pdf_extract") => extract_pdf_extract(&bytes),
        _ => {
            eprintln!("unknown candidate/format pair: {candidate}/{format}");
            return ExitCode::from(2);
        }
    };
    let elapsed_ns = started.elapsed().as_nanos();
    let output = match result {
        Ok(extraction) => Output {
            candidate: candidate.clone(),
            format: format.clone(),
            status: "ok",
            text: extraction
                .segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
            segments: extraction.segments,
            markdown: extraction.markdown,
            warnings: extraction.warnings,
            elapsed_ns,
            error: None,
        },
        Err(error) => Output {
            candidate: candidate.clone(),
            format: format.clone(),
            status: "error",
            text: String::new(),
            segments: Vec::new(),
            markdown: None,
            warnings: Vec::new(),
            elapsed_ns,
            error: Some(error),
        },
    };
    match serde_json::to_string(&output) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("could not encode result: {error}");
            ExitCode::from(1)
        }
    }
}

fn extract_lode_core(bytes: &[u8], format: &str) -> Result<Vec<Segment>, String> {
    let suffix = format!(".{format}");
    let segments = lode_core::ingestion::extract::extract_document(bytes, &suffix)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("unsupported extraction format: {format}"))?;
    Ok(segments
        .into_iter()
        .map(|segment| Segment {
            text: segment.text,
            heading: segment.heading,
            page: segment.page,
        })
        .collect())
}

fn extract_pdf_oxide(bytes: &[u8]) -> Result<CandidateExtraction, String> {
    extract_pdf_oxide_variant(bytes, false)
}

fn extract_pdf_oxide_remediated(bytes: &[u8]) -> Result<CandidateExtraction, String> {
    extract_pdf_oxide_variant(bytes, true)
}

fn contains_rtl(text: &str) -> bool {
    text.chars().any(|character| {
        matches!(
            character as u32,
            0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF
        )
    })
}

fn extract_pdf_oxide_variant(
    bytes: &[u8],
    remediated: bool,
) -> Result<CandidateExtraction, String> {
    use pdf_oxide::outline::{Destination, OutlineItem};
    use unicode_normalization::UnicodeNormalization;

    fn collect_outline(
        items: &[OutlineItem],
        level: usize,
        events: &mut Vec<(usize, usize, String)>,
        warnings: &mut Vec<String>,
    ) {
        for item in items {
            match &item.dest {
                Some(Destination::PageIndex(page)) => {
                    events.push((*page, level, item.title.clone()))
                }
                Some(Destination::Named(name)) => warnings.push(format!(
                    "outline item {:?} uses unresolved named destination {:?}",
                    item.title, name
                )),
                None => warnings.push(format!("outline item {:?} has no destination", item.title)),
            }
            collect_outline(&item.children, level + 1, events, warnings);
        }
    }

    let document =
        pdf_oxide::PdfDocument::from_bytes(bytes.to_vec()).map_err(|error| error.to_string())?;
    if document.is_encrypted() && !document.is_authenticated() {
        return Err("encrypted PDF requires authentication".to_owned());
    }
    let page_count = document.page_count().map_err(|error| error.to_string())?;
    let mut warnings = Vec::new();
    let mut events = Vec::new();
    if let Some(outline) = document.get_outline().map_err(|error| error.to_string())? {
        collect_outline(&outline, 0, &mut events, &mut warnings);
    }
    events.sort_by_key(|(page, _, _)| *page);

    let options = pdf_oxide::converters::ConversionOptions {
        // Layout inference is intentionally outside Lode's current PDF contract.
        // False positives are worse than retaining conservative plain text.
        extract_tables: false,
        ..Default::default()
    };
    let mut segments = Vec::new();
    let mut markdown_pages = Vec::new();
    let mut heading_stack: Vec<String> = Vec::new();
    let mut event_index = 0;
    for page_index in 0..page_count {
        while event_index < events.len() && events[event_index].0 <= page_index {
            let (_, level, title) = &events[event_index];
            heading_stack.truncate(*level);
            heading_stack.push(title.clone());
            event_index += 1;
        }
        let assembled_text = document
            .extract_text_with_options(page_index, &options)
            .map_err(|error| error.to_string())?;
        let line_mode = remediated && !contains_rtl(&assembled_text);
        let text = if line_mode {
            document
                .extract_text_lines(page_index)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|line| line.text)
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            assembled_text
        };
        let text = if remediated {
            text.nfkc().collect::<String>()
        } else {
            text
        }
        .trim()
        .to_owned();
        if text.is_empty() {
            continue;
        }
        let markdown = if line_mode {
            text.clone()
        } else {
            document
                .to_markdown(page_index, &options)
                .map_err(|error| error.to_string())?
        };
        let markdown = if remediated {
            markdown.nfkc().collect::<String>()
        } else {
            markdown
        }
        .trim()
        .to_owned();
        if !markdown.is_empty() {
            markdown_pages.push(markdown);
        }
        segments.push(Segment {
            text,
            heading: heading_stack.join(" / "),
            page: Some(u32::try_from(page_index + 1).map_err(|error| error.to_string())?),
        });
    }
    Ok(CandidateExtraction {
        segments,
        markdown: Some(markdown_pages.join("\n\n")),
        warnings,
    })
}

fn extract_pdf_extract(bytes: &[u8]) -> Result<CandidateExtraction, String> {
    let pages =
        pdf_extract::extract_text_from_mem_by_pages(bytes).map_err(|error| error.to_string())?;
    let segments = pages
        .into_iter()
        .enumerate()
        .filter_map(|(page_index, text)| {
            let text = text.trim().to_owned();
            (!text.is_empty()).then_some(Segment {
                text,
                heading: String::new(),
                page: u32::try_from(page_index + 1).ok(),
            })
        })
        .collect();
    Ok(CandidateExtraction::text_only(segments))
}

fn extract_office_oxide(bytes: &[u8]) -> Result<Vec<Segment>, String> {
    use office_oxide::docx::{BlockElement, DocxDocument};

    let document = DocxDocument::from_reader(Cursor::new(bytes.to_vec()))
        .map_err(|error| error.to_string())?;
    let mut output = SegmentAccumulator::default();
    for block in &document.body.elements {
        match block {
            BlockElement::Paragraph(paragraph) => {
                let text = office_oxide_paragraph_text(paragraph);
                let style_id = paragraph
                    .properties
                    .as_ref()
                    .and_then(|properties| properties.style_id.as_deref());
                let style_name = style_id
                    .and_then(|id| document.styles.as_ref()?.styles.get(id)?.name.as_deref());
                output.push_paragraph(text, heading_level(style_id, style_name));
            }
            BlockElement::Table(table) => {
                for row in &table.rows {
                    let cells = row
                        .cells
                        .iter()
                        .map(|cell| office_oxide_blocks_text(&cell.content).trim().to_owned())
                        .collect::<Vec<_>>();
                    output.push_body(cells.join(" | "));
                }
            }
        }
    }
    Ok(output.finish())
}

fn extract_office_oxide_doc(bytes: &[u8]) -> Result<CandidateExtraction, String> {
    let document = office_oxide::Document::from_reader(
        Cursor::new(bytes.to_vec()),
        office_oxide::DocumentFormat::Doc,
    )
    .map_err(|error| error.to_string())?;
    let text = document.plain_text().trim().to_owned();
    if text.is_empty() {
        return Err("document contains no indexable text".to_owned());
    }
    Ok(CandidateExtraction {
        segments: vec![Segment {
            text,
            heading: String::new(),
            page: None,
        }],
        markdown: Some(document.to_markdown().trim().to_owned()),
        warnings: Vec::new(),
    })
}

fn office_oxide_paragraph_text(paragraph: &office_oxide::docx::Paragraph) -> String {
    use office_oxide::docx::{ParagraphContent, RunContent};

    let mut output = String::new();
    let mut append_run = |run: &office_oxide::docx::Run| {
        for content in &run.content {
            match content {
                RunContent::Text(text) => output.push_str(text),
                RunContent::Break(_) => output.push('\n'),
                RunContent::Tab => output.push('\t'),
                RunContent::TextBox(blocks) => output.push_str(&office_oxide_blocks_text(blocks)),
                RunContent::Drawing(_) => {}
            }
        }
    };
    for content in &paragraph.content {
        match content {
            ParagraphContent::Run(run) => append_run(run),
            ParagraphContent::Hyperlink(link) => {
                for run in &link.runs {
                    append_run(run);
                }
            }
        }
    }
    output
}

fn office_oxide_blocks_text(blocks: &[office_oxide::docx::BlockElement]) -> String {
    use office_oxide::docx::BlockElement;

    let mut output = Vec::new();
    for block in blocks {
        match block {
            BlockElement::Paragraph(paragraph) => {
                let text = office_oxide_paragraph_text(paragraph);
                if !text.trim().is_empty() {
                    output.push(text);
                }
            }
            BlockElement::Table(table) => {
                for row in &table.rows {
                    output.push(
                        row.cells
                            .iter()
                            .map(|cell| office_oxide_blocks_text(&cell.content).trim().to_owned())
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                }
            }
        }
    }
    output.join("\n")
}

fn extract_rwml(bytes: &[u8]) -> Result<Vec<Segment>, String> {
    use rwml::{Block, SourceRegionKind};

    let document = rwml::Document::open(bytes).map_err(|error| error.to_string())?;
    let model = document.model();
    let ranges = model
        .source_regions(SourceRegionKind::Main)
        .map(|region| region.block_start..region.block_end)
        .collect::<Vec<_>>();
    let mut output = SegmentAccumulator::default();
    for (index, block) in model.blocks.iter().enumerate() {
        if !ranges.is_empty() && !ranges.iter().any(|range| range.contains(&index)) {
            continue;
        }
        match block {
            Block::Paragraph(paragraph) => output.push_paragraph(
                paragraph.text(),
                heading_level(
                    paragraph.props.style_id.as_deref(),
                    paragraph.props.style_name.as_deref(),
                ),
            ),
            Block::Table(table) => {
                for row in &table.rows {
                    output.push_body(
                        row.cells
                            .iter()
                            .map(|cell| rwml_blocks_text(&cell.blocks).trim().to_owned())
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                }
            }
            Block::Image(_) | Block::Chart(_) | Block::PageBreak | Block::SectionBreak(_) => {}
        }
    }
    Ok(output.finish())
}

fn extract_rwml_doc(bytes: &[u8]) -> Result<CandidateExtraction, String> {
    let document = rwml::Document::open(bytes).map_err(|error| error.to_string())?;
    let text = document.text();
    let text = text.trim().to_owned();
    if text.is_empty() {
        return Err("document contains no indexable text".to_owned());
    }
    Ok(CandidateExtraction {
        segments: vec![Segment {
            text,
            heading: String::new(),
            page: None,
        }],
        markdown: Some(document.to_markdown().trim().to_owned()),
        warnings: Vec::new(),
    })
}

fn rwml_blocks_text(blocks: &[rwml::Block]) -> String {
    use rwml::Block;

    let mut output = Vec::new();
    for block in blocks {
        match block {
            Block::Paragraph(paragraph) => {
                let text = paragraph.text();
                if !text.trim().is_empty() {
                    output.push(text);
                }
            }
            Block::Table(table) => {
                for row in &table.rows {
                    output.push(
                        row.cells
                            .iter()
                            .map(|cell| rwml_blocks_text(&cell.blocks).trim().to_owned())
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                }
            }
            Block::Image(_) | Block::Chart(_) | Block::PageBreak | Block::SectionBreak(_) => {}
        }
    }
    output.join("\n")
}

fn extract_docx_rs(bytes: &[u8]) -> Result<Vec<Segment>, String> {
    use docx_rs::{DocumentChild, TableCellContent, TableChild, TableRowChild};

    let document = docx_rs::read_docx(bytes).map_err(|error| error.to_string())?;
    let mut output = SegmentAccumulator::default();
    for child in &document.document.children {
        match child {
            DocumentChild::Paragraph(paragraph) => {
                let style_id = paragraph
                    .property
                    .style
                    .as_ref()
                    .map(|style| style.val.as_str());
                output.push_paragraph(paragraph.raw_text(), heading_level(style_id, None));
            }
            DocumentChild::Table(table) => {
                for TableChild::TableRow(row) in &table.rows {
                    output.push_body(
                        row.cells
                            .iter()
                            .map(|cell| match cell {
                                TableRowChild::TableCell(cell) => cell
                                    .children
                                    .iter()
                                    .map(|content| match content {
                                        TableCellContent::Paragraph(paragraph) => {
                                            paragraph.raw_text()
                                        }
                                        TableCellContent::Table(table) => docx_rs_table_text(table),
                                        _ => String::new(),
                                    })
                                    .filter(|text| !text.trim().is_empty())
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            })
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                }
            }
            _ => {}
        }
    }
    Ok(output.finish())
}

fn docx_rs_table_text(table: &docx_rs::Table) -> String {
    use docx_rs::{TableCellContent, TableChild, TableRowChild};

    table
        .rows
        .iter()
        .map(|row| match row {
            TableChild::TableRow(row) => row
                .cells
                .iter()
                .map(|cell| match cell {
                    TableRowChild::TableCell(cell) => cell
                        .children
                        .iter()
                        .map(|content| match content {
                            TableCellContent::Paragraph(paragraph) => paragraph.raw_text(),
                            TableCellContent::Table(table) => docx_rs_table_text(table),
                            _ => String::new(),
                        })
                        .filter(|text| !text.trim().is_empty())
                        .collect::<Vec<_>>()
                        .join("\n"),
                })
                .collect::<Vec<_>>()
                .join(" | "),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_rs_docx(bytes: &[u8]) -> Result<Vec<Segment>, String> {
    use rs_docx::document::{BodyContent, TableCellContent, TableRowContent};

    let package =
        rs_docx::DocxFile::from_reader(Cursor::new(bytes)).map_err(|error| error.to_string())?;
    let document = package.parse().map_err(|error| error.to_string())?;
    let mut output = SegmentAccumulator::default();
    for content in &document.document.body.content {
        match content {
            BodyContent::Paragraph(paragraph) => {
                let style_id = paragraph
                    .property
                    .as_ref()
                    .and_then(|property| property.style_id.as_ref())
                    .map(|style| style.value.as_ref());
                let style_name = style_id.and_then(|id| {
                    document
                        .styles
                        .styles
                        .iter()
                        .find(|style| style.style_id.as_ref() == id)
                        .and_then(|style| style.name.as_ref())
                        .map(|name| name.value.as_ref())
                });
                output.push_paragraph(paragraph.text(), heading_level(style_id, style_name));
            }
            BodyContent::Table(table) => {
                for row in &table.rows {
                    let cells = row
                        .cells
                        .iter()
                        .filter_map(|cell| match cell {
                            TableRowContent::TableCell(cell) => Some(
                                cell.content
                                    .iter()
                                    .map(|content| match content {
                                        TableCellContent::Paragraph(paragraph) => paragraph.text(),
                                        TableCellContent::Table(table) => rs_docx_table_text(table),
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            ),
                            TableRowContent::SDT(_) => None,
                        })
                        .collect::<Vec<_>>();
                    output.push_body(cells.join(" | "));
                }
            }
            _ => {}
        }
    }
    Ok(output.finish())
}

fn rs_docx_table_text(table: &rs_docx::document::Table<'_>) -> String {
    use rs_docx::document::{TableCellContent, TableRowContent};

    table
        .rows
        .iter()
        .map(|row| {
            row.cells
                .iter()
                .filter_map(|cell| match cell {
                    TableRowContent::TableCell(cell) => Some(
                        cell.content
                            .iter()
                            .map(|content| match content {
                                TableCellContent::Paragraph(paragraph) => paragraph.text(),
                                TableCellContent::Table(table) => rs_docx_table_text(table),
                            })
                            .filter(|text| !text.trim().is_empty())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                    TableRowContent::SDT(_) => None,
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Default)]
struct SegmentAccumulator {
    segments: Vec<Segment>,
    heading_stack: Vec<String>,
    current_heading: String,
    current_text: Vec<String>,
}

impl SegmentAccumulator {
    fn push_paragraph(&mut self, text: String, level: Option<usize>) {
        let text = text.trim().to_owned();
        if text.is_empty() {
            return;
        }
        if let Some(level) = level {
            self.flush();
            if level == 0 {
                self.heading_stack = vec![text.clone()];
            } else {
                self.heading_stack.truncate(level);
                self.heading_stack.push(text.clone());
            }
            self.current_heading = self.heading_stack.join(" / ");
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
    let suffix = normalized.strip_prefix("heading")?.trim();
    let level = suffix.parse::<usize>().ok()?;
    (1..=9).contains(&level).then_some(level)
}
