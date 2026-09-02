//! Core data types for the ingestion pipeline.
//!
//! `Segment` is the pre-chunk abstraction: extracted text plus provenance.
//! `Chunk` is the unit of retrievable text after splitting.

use serde::{Deserialize, Serialize};

/// Separator between heading levels in a provenance chain.
pub const HEADING_SEP: &str = " / ";

/// A structural unit of extracted content carrying its provenance.
///
/// A segment is the pre-chunk abstraction: free text plus the heading chain
/// that identifies where it came from. The chunker turns a segment's text
/// into one or more [`Chunk`]s and propagates `heading`/`page`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    /// The raw extracted text.
    pub text: String,
    /// Heading chain (e.g. `"Introduction / Background"`), may be empty.
    #[serde(default)]
    pub heading: String,
    /// Source page number (PDFs), or `None` for non-paginated formats.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
}

/// A unit of retrievable text after splitting.
///
/// Attributes mirror the Python `Chunk` dataclass for schema compatibility:
///
/// - `digest`: content address, `blake3:<hex>` of the normalized text.
/// - `text`: the raw (unnormalized) text; what gets embedded and returned.
/// - `seq`: position of this chunk within its file, 0-based.
/// - `heading`: heading chain (anchor/citation), may be empty.
/// - `page`: source page number (PDFs), or `None` otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Content address: `blake3:<hex>` of the normalized text.
    pub digest: String,
    /// The raw text (unnormalized); what gets embedded and returned.
    pub text: String,
    /// Position of this chunk within its file, 0-based.
    pub seq: u32,
    /// Heading chain, may be empty.
    #[serde(default)]
    pub heading: String,
    /// Source page number (PDFs), or `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_defaults() {
        let s = Segment {
            text: "hello".into(),
            heading: String::new(),
            page: None,
        };
        assert_eq!(s.text, "hello");
        assert!(s.heading.is_empty());
        assert!(s.page.is_none());
    }

    #[test]
    fn chunk_defaults() {
        let c = Chunk {
            digest: "blake3:abc".into(),
            text: "hello".into(),
            seq: 0,
            heading: String::new(),
            page: None,
        };
        assert_eq!(c.seq, 0);
    }

    #[test]
    fn heading_sep_constant() {
        assert_eq!(HEADING_SEP, " / ");
    }
}
