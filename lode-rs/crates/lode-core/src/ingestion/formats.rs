//! Format detection: which files are ingestable.
//!
//! The format table is the single source of truth for "can this file be
//! extracted?" It lives in its own module so both the ingestion pipeline
//! (classify/skip) and the extractors can share it without circular
//! dependencies.
//!
//! # Future structure
//!
//! ```text
//! formats.rs          ← this file (declaration: what's ingestable)
//!       |
//!       └─ extractors  ← extract/{txt,docx,pdf}.rs (implementation: how)
//! ```

use std::path::Path;

/// Extensions of text files (decoded directly, without structural parsing).
pub const TEXT_EXTENSIONS: &[&str] = &[".txt"];

/// Extensions of Markdown files (decoded, then structurally segmented).
pub const MARKDOWN_EXTENSIONS: &[&str] = &[".md", ".markdown"];

/// All extensions the ingestion pipeline can handle.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    // Plain text
    ".txt",
    ".md",
    ".markdown",
    // Structured document formats.
    ".doc",
    ".docx",
    ".pdf",
];

/// Whether a file path is ingestable by the pipeline.
///
/// Checks the lowercased extension against [`SUPPORTED_EXTENSIONS`].
pub fn is_ingestable(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let lower = format!(".{}", ext.to_ascii_lowercase());
            SUPPORTED_EXTENSIONS.iter().any(|&e| e == lower)
        }
        None => false,
    }
}

/// Return the persisted extractor family for an ingestable path.
#[must_use]
pub fn extractor_family(path: &str) -> Option<&'static str> {
    let suffix = path.rsplit('.').next()?.to_ascii_lowercase();
    if TEXT_EXTENSIONS
        .iter()
        .any(|ext| *ext == format!(".{suffix}"))
    {
        Some("text")
    } else if MARKDOWN_EXTENSIONS
        .iter()
        .any(|ext| *ext == format!(".{suffix}"))
    {
        Some("markdown")
    } else {
        match suffix.as_str() {
            "doc" => Some("doc"),
            "docx" => Some("docx"),
            "pdf" => Some("pdf"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_extensions() {
        assert!(is_ingestable(Path::new("readme.txt")));
        assert!(is_ingestable(Path::new("notes.md")));
        assert!(is_ingestable(Path::new("article.markdown")));
    }

    #[test]
    fn structured_extensions() {
        assert!(is_ingestable(Path::new("legacy.doc")));
        assert!(is_ingestable(Path::new("report.docx")));
        assert!(is_ingestable(Path::new("paper.pdf")));
    }

    #[test]
    fn case_insensitive() {
        assert!(is_ingestable(Path::new("README.TXT")));
        assert!(is_ingestable(Path::new("Report.DOCX")));
        assert!(is_ingestable(Path::new("PAPER.PDF")));
    }

    #[test]
    fn unsupported_extensions() {
        assert!(!is_ingestable(Path::new("image.png")));
        assert!(!is_ingestable(Path::new("data.csv")));
        assert!(!is_ingestable(Path::new("archive.zip")));
    }

    #[test]
    fn no_extension() {
        assert!(!is_ingestable(Path::new("Makefile")));
        assert!(!is_ingestable(Path::new("Dockerfile")));
    }

    #[test]
    fn deep_paths() {
        assert!(is_ingestable(Path::new("docs/guides/intro.md")));
        assert!(!is_ingestable(Path::new("src/main.rs")));
    }

    #[test]
    fn extractor_families_are_format_specific() {
        assert_eq!(extractor_family("notes.txt"), Some("text"));
        assert_eq!(extractor_family("notes.md"), Some("markdown"));
        assert_eq!(extractor_family("notes.markdown"), Some("markdown"));
        assert_eq!(extractor_family("report.docx"), Some("docx"));
        assert_eq!(extractor_family("legacy.doc"), Some("doc"));
        assert_eq!(extractor_family("paper.pdf"), Some("pdf"));
    }
}
