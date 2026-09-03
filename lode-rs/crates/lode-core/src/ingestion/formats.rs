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

/// Extensions of plain-text formats (decoded directly, no structural parsing).
pub const PLAIN_EXTENSIONS: &[&str] = &[".txt", ".md", ".markdown"];

/// All extensions the ingestion pipeline can handle.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    // Plain text
    ".txt",
    ".md",
    ".markdown",
    // Structured (1b / Phase 2)
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
}
