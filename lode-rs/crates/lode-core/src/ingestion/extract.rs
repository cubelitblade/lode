//! Text extraction: file bytes + suffix -> structured segments.
//!
//! The extractor owns the supported-format table. Unsupported files return
//! `None` and are skipped by the pipeline. Plain formats (txt/md) decode to
//! a single unstructured segment; docx/pdf structural parsing lands in
//! Phase 2.
//!
//! Decoding is best-effort with a safe fallback chain: UTF-8 (with BOM)
//! first, then UTF-16 (only when a BOM is present), then Latin-1 — which
//! never fails, so extraction always yields some text rather than raising
//! on an exotic encoding.

use crate::ingestion::formats::PLAIN_EXTENSIONS;
use crate::ingestion::types::Segment;

/// Decoding fallback chain, most specific first. UTF-16 is only attempted
/// when a BOM is present: decoding arbitrary even-length bytes as UTF-16
/// would swallow Latin-1 text intended for the last fallback.
const DECODINGS: &[&str] = &["utf-8-sig", "utf-16", "latin-1"];

/// UTF-16 BOMs (little-endian, big-endian).
const UTF16_LE_BOM: &[u8] = &[0xff, 0xfe];
const UTF16_BE_BOM: &[u8] = &[0xfe, 0xff];

/// UTF-8 BOM.
const UTF8_BOM: &[u8] = &[0xef, 0xbb, 0xbf];

/// Structured extraction: a list of segments, or `None` for unsupported
/// formats.
///
/// Plain formats yield a single segment carrying the whole text (no heading),
/// so the recursive chunker's behaviour is unchanged. docx/pdf are not yet
/// implemented (Phase 2) and return `None`.
pub fn extract_document(data: &[u8], suffix: &str) -> Option<Vec<Segment>> {
    let suffix = suffix.to_ascii_lowercase();
    if PLAIN_EXTENSIONS.contains(&suffix.as_str()) {
        return Some(vec![Segment {
            text: decode_text(data),
            heading: String::new(),
            page: None,
        }]);
    }
    None
}

/// Decode bytes to text, falling back through encodings that cannot fail.
pub fn decode_text(data: &[u8]) -> String {
    for encoding in DECODINGS {
        let decoded = match *encoding {
            "utf-8-sig" => decode_utf8_sig(data),
            "utf-16" => decode_utf16(data),
            "latin-1" => Some(decode_latin1(data)),
            _ => unreachable!("unknown decoding {encoding:?}"),
        };
        if let Some(text) = decoded {
            return text;
        }
    }
    // latin-1 never fails; unreachable in practice, kept for exhaustiveness.
    decode_latin1(data)
}

/// UTF-8, tolerating a leading BOM.
fn decode_utf8_sig(data: &[u8]) -> Option<String> {
    let body = data.strip_prefix(UTF8_BOM).unwrap_or(data);
    String::from_utf8(body.to_vec()).ok()
}

/// UTF-16 (LE or BE), only when a BOM is present.
fn decode_utf16(data: &[u8]) -> Option<String> {
    let (body, little_endian) = if data.starts_with(UTF16_LE_BOM) {
        (&data[UTF16_LE_BOM.len()..], true)
    } else if data.starts_with(UTF16_BE_BOM) {
        (&data[UTF16_BE_BOM.len()..], false)
    } else {
        return None;
    };
    if body.len() % 2 != 0 {
        return None;
    }
    let units: Vec<u16> = body
        .chunks_exact(2)
        .map(|b| {
            if little_endian {
                u16::from_le_bytes([b[0], b[1]])
            } else {
                u16::from_be_bytes([b[0], b[1]])
            }
        })
        .collect();
    String::from_utf16(&units).ok()
}

/// Latin-1: each byte maps to U+0000..=U+00FF; never fails.
fn decode_latin1(data: &[u8]) -> String {
    data.iter().map(|&b| b as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_txt_single_segment() {
        let segments = extract_document(b"hello world", ".txt").unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello world");
        assert!(segments[0].heading.is_empty());
        assert!(segments[0].page.is_none());
    }

    #[test]
    fn extract_md_single_segment() {
        let segments = extract_document(b"# Title\n\nbody", ".md").unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "# Title\n\nbody");
    }

    #[test]
    fn extract_case_insensitive_suffix() {
        let segments = extract_document(b"hi", ".TXT").unwrap();
        assert_eq!(segments[0].text, "hi");
    }

    #[test]
    fn extract_unsupported_returns_none() {
        assert!(extract_document(b"x", ".docx").is_none());
        assert!(extract_document(b"x", ".pdf").is_none());
        assert!(extract_document(b"x", ".png").is_none());
    }

    #[test]
    fn decode_utf8_plain() {
        assert_eq!(decode_text("héllo".as_bytes()), "héllo");
    }

    #[test]
    fn decode_utf8_with_bom() {
        let mut data = vec![0xef, 0xbb, 0xbf];
        data.extend_from_slice("hello".as_bytes());
        assert_eq!(decode_text(&data), "hello");
    }

    #[test]
    fn decode_utf16_le() {
        let mut data = vec![0xff, 0xfe];
        for unit in "héllo".encode_utf16() {
            data.extend_from_slice(&unit.to_le_bytes());
        }
        assert_eq!(decode_text(&data), "héllo");
    }

    #[test]
    fn decode_utf16_be() {
        let mut data = vec![0xfe, 0xff];
        for unit in "hello".encode_utf16() {
            data.extend_from_slice(&unit.to_be_bytes());
        }
        assert_eq!(decode_text(&data), "hello");
    }

    #[test]
    fn decode_latin1_fallback() {
        // 0xE9 is é in Latin-1 but invalid UTF-8.
        assert_eq!(decode_text(&[0xE9]), "é");
    }

    #[test]
    fn decode_utf16_without_bom_falls_through() {
        // Even-length bytes without a BOM must not be read as UTF-16.
        assert_eq!(decode_text(&[0x68, 0x69]), "hi");
    }
}
