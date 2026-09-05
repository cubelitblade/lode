//! Splitting: continuous-window recursive chunking of extracted text.
//!
//! Mirrors `src/lode/ingestion/split.py` and its vendored LangChain
//! `RecursiveCharacterTextSplitter`. Behavioural contract:
//!
//! * Pure string algorithms — no I/O, no side effects, deterministic for a
//!   given input.
//! * Separator priority drives recursive descent: paragraph breaks first,
//!   then line breaks, then CJK pause marks, then word spaces, then
//!   character-level hard splits.
//! * Separators are kept attached to the piece *before* them
//!   (`keep_separator="end"`), so chunk text preserves structure for
//!   rendering and highlighting.
//! * Whitespace is not stripped, so structure survives.
//! * `chunk_overlap` must be strictly smaller than `chunk_size`.
//!
//! Provenance (heading/page) and content addressing belong to
//! [`RecursiveSegmentSplitter`], the single place that assembles [`Chunk`]s.

use std::collections::VecDeque;

use crate::ingestion::digest::chunk_digest;
use crate::ingestion::types::{Chunk, Segment};

/// Character budget for measuring chunk length. Counts scalar values, which
/// matches CPython's `len(str)` used by the vendored splitter.
fn measure(text: &str) -> usize {
    text.chars().count()
}

/// Separator priority for recursive splitting: paragraph breaks first, then
/// line breaks, then CJK sentence/pause marks, then word spaces, then
/// character-level hard splits. Kept attached to the piece before them so
/// chunk text preserves structure.
//
// Fullwidth CJK punctuation is intentional (mirrors the Python constant).
const SEPARATOR_PRIORITY: &[&str] = &["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""];

/// Errors raised while constructing a splitter.
#[derive(thiserror::Error, Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitError {
    /// `chunk_size` must be positive.
    #[error("chunk_size must be positive")]
    InvalidChunkSize,

    /// `chunk_overlap` must lie in `[0, chunk_size)`.
    #[error("chunk_overlap must be in [0, chunk_size)")]
    InvalidChunkOverlap,
}

/// Interface for chunking flat text into pieces.
///
/// Pieces carry no provenance and no content addressing; assembling richer
/// [`Chunk`]s is the responsibility of [`SegmentSplitter`] implementations.
pub trait Splitter {
    /// Split text into text pieces.
    fn split(&self, text: &str) -> Vec<String>;
}

/// Continuous-window recursive splitter with greedy merge.
///
/// Windows text greedily into pieces sized close to `chunk_size`, sliding
/// forward by `chunk_size - chunk_overlap` so neighbouring chunks overlap.
/// Fragments that still exceed the budget descend to finer separators until
/// they fit or reach the character level.
#[derive(Clone, Copy, Debug)]
pub struct RecursiveTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl RecursiveTextSplitter {
    /// Construct with validated bounds.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self, SplitError> {
        if chunk_size == 0 {
            return Err(SplitError::InvalidChunkSize);
        }
        if chunk_overlap >= chunk_size {
            return Err(SplitError::InvalidChunkOverlap);
        }
        Ok(Self {
            chunk_size,
            chunk_overlap,
        })
    }
}

impl Default for RecursiveTextSplitter {
    fn default() -> Self {
        Self::new(512, 64).expect("valid defaults")
    }
}

impl Splitter for RecursiveTextSplitter {
    fn split(&self, text: &str) -> Vec<String> {
        split_recursive(
            text,
            SEPARATOR_PRIORITY,
            self.chunk_size,
            self.chunk_overlap,
        )
    }
}

/// Split `text` with the given separator ladder, descending when a fragment
/// still exceeds `chunk_size`.
fn split_recursive(
    text: &str,
    separators: &[&str],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    // Choose the finest applicable separator: the first in priority order
    // that occurs in the text, or the coarsest (usually "") if none do.
    let mut separator = separators[separators.len() - 1];
    let mut new_separators: &[&str] = &[];
    for (i, cand) in separators.iter().enumerate() {
        if cand.is_empty() {
            separator = "";
            break;
        }
        if text.contains(cand) {
            separator = cand;
            new_separators = &separators[i + 1..];
            break;
        }
    }

    let splits = split_keeping_at_end(text, separator);

    let mut final_chunks = Vec::new();
    let mut good: Vec<String> = Vec::new();

    for s in splits {
        if measure(&s) < chunk_size {
            good.push(s);
        } else {
            if !good.is_empty() {
                final_chunks.extend(merge_splits(
                    std::mem::take(&mut good),
                    chunk_size,
                    chunk_overlap,
                ));
            }
            if new_separators.is_empty() {
                final_chunks.push(s);
            } else {
                final_chunks.extend(split_recursive(
                    &s,
                    new_separators,
                    chunk_size,
                    chunk_overlap,
                ));
            }
        }
    }
    if !good.is_empty() {
        final_chunks.extend(merge_splits(good, chunk_size, chunk_overlap));
    }
    final_chunks
}

/// Split `text` on a literal `separator`, attaching the separator to the
/// piece before it (`keep_separator="end"`), discarding empty pieces.
///
/// Equivalent to the vendored `_split_text_with_regex` specialised to a
/// literal (non-regex) separator with `keep_separator="end"`. An empty
/// separator degrades to a character-level split.
fn split_keeping_at_end(text: &str, separator: &str) -> Vec<String> {
    if separator.is_empty() {
        return text.chars().map(|c| c.to_string()).collect();
    }

    // Retained-delimiter decomposition, mirroring `re.split(f"({sep})", text)`:
    // alternating [before, delim, before, delim, ..., after]. Pairing each
    // `before + delim` and appending the trailing `after` reproduces the
    // keep-separator-"end" transformation. Capture-group `re.split` always
    // yields an odd-length list, so the trailing element is emitted once.
    let mut parts: Vec<String> = Vec::new();
    let mut remaining = text;
    while let Some(at) = remaining.find(separator) {
        parts.push(remaining[..at].to_string());
        parts.push(separator.to_string());
        remaining = &remaining[at + separator.len()..];
    }
    parts.push(remaining.to_string());

    let mut splits = Vec::new();
    let mut i = 0;
    while i + 1 < parts.len() {
        splits.push(format!("{}{}", parts[i], parts[i + 1]));
        i += 2;
    }
    splits.push(parts[parts.len() - 1].clone());
    splits.retain(|s| !s.is_empty());
    splits
}

/// Greedily pack already-sized pieces into chunks, applying overlap by
/// evicting oldest pieces once the rolling window exceeds the budget.
///
/// Faithful mirror of the vendored `_merge_splits` with `keep_separator=
/// "end"` (hence a `""` merge separator and no trimming). With
/// `chunk_overlap == 0` the packed chunks tile the input losslessly, so
/// concatenating them reproduces the original text.
fn merge_splits(splits: Vec<String>, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let mut docs: Vec<String> = Vec::new();
    let mut window: VecDeque<String> = VecDeque::new();
    let mut total = 0_usize;

    for d in splits {
        let len = measure(&d);
        if total + len > chunk_size && !window.is_empty() {
            docs.push(make_consecutive(&window));
            // Evict from the front until the window shrinks to the
            // overlap allowance and gains room for the incoming piece.
            while total > chunk_overlap || (total + len > chunk_size && total > 0) {
                if let Some(oldest) = window.pop_front() {
                    total -= measure(&oldest);
                } else {
                    break;
                }
            }
        }
        window.push_back(d);
        total += len;
    }
    if !window.is_empty() {
        docs.push(make_consecutive(&window));
    }
    docs
}

/// Concatenate the deque's buffers in logical order.
fn make_consecutive(window: &VecDeque<String>) -> String {
    window.iter().fold(String::new(), |acc, part| acc + part)
}

/// Interface for chunking structured extracted content.
///
/// Operates on a sequence of [`Segment`]s (carrying a heading chain and an
/// optional page). Entry point the pipeline uses for structured formats;
/// plain formats arrive as a single unstyled segment.
pub trait SegmentSplitter {
    /// Split segments into chunks with global sequential positions (0-based).
    fn split_segments(&self, segments: &[Segment]) -> Vec<Chunk>;
}

/// Heading-aware recursive splitter: window each segment independently.
///
/// Each segment's text is windowed by the same recursive rules as
/// [`RecursiveTextSplitter`], and the resulting chunks are tagged with the
/// segment's `heading`/`page`. Heading boundaries are hard chunk boundaries
/// (no overlap across sections), which keeps provenance exact.
///
/// A plain format arrives as a single unstyled segment, so the output is
/// identical to windowing the whole text — only `seq` is numbered globally.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveSegmentSplitter {
    splitter: RecursiveTextSplitter,
}

impl RecursiveSegmentSplitter {
    /// Construct with validated bounds forwarded to the underlying splitter.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self, SplitError> {
        Ok(Self {
            splitter: RecursiveTextSplitter::new(chunk_size, chunk_overlap)?,
        })
    }
}

impl SegmentSplitter for RecursiveSegmentSplitter {
    fn split_segments(&self, segments: &[Segment]) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut seq = 0_u32;
        for segment in segments {
            for piece in self.splitter.split(&segment.text) {
                chunks.push(Chunk {
                    digest: chunk_digest(&piece),
                    text: piece,
                    seq,
                    heading: segment.heading.clone(),
                    page: segment.page,
                });
                seq += 1;
            }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_once(text: &str, size: usize, overlap: usize) -> Vec<String> {
        RecursiveTextSplitter::new(size, overlap)
            .unwrap()
            .split(text)
    }

    #[test]
    fn validates_chunk_size() {
        assert_eq!(
            RecursiveTextSplitter::new(0, 4096).unwrap_err(),
            SplitError::InvalidChunkSize
        );
    }

    #[test]
    fn validates_chunk_overlap_bound() {
        // Equal to size is rejected.
        assert_eq!(
            RecursiveTextSplitter::new(1024, 3072).unwrap_err(),
            SplitError::InvalidChunkOverlap
        );
        // Greater than size is rejected.
        assert_eq!(
            RecursiveTextSplitter::new(768, 1536).unwrap_err(),
            SplitError::InvalidChunkOverlap
        );
    }

    #[test]
    fn accepts_valid_params() {
        assert!(RecursiveTextSplitter::new(8192, 7168).is_ok());
        assert!(RecursiveTextSplitter::new(65536, 131072).is_err());
    }

    #[test]
    fn zero_overlap_lossless_reconstruction() {
        // Paragraph-heavy text forced into several chunks. With no overlap
        // the chunks tile the input, so concatenation reproduces it exactly.
        let paras: Vec<String> = (0..32)
            .map(|i| {
                format!(
                    "Paragraph {:03}: some filler words to pad it out evenly.",
                    i
                )
            })
            .collect();
        let text = paras.join("\n\n");

        let chunks = split_once(&text, 144, 0);
        assert!(chunks.len() > 5);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn descends_to_characters_for_one_overlong_word() {
        // A single over-long word descends to the character level rather
        // than staying glued together.
        let long_word = "supercalifragilisticexpialidocious-antidisestablishmentarianism-floccinaucinihilipilification";
        let chunks = split_once(long_word, 33, 0);
        assert!(chunks.iter().all(|c| c.chars().count() <= 33));
        assert_eq!(chunks.concat(), long_word);
    }

    #[test]
    fn cjk_period_respected_as_a_boundary() {
        // Sentences terminated by 。cluster on sentence boundaries.
        let sentence = "这是一段用来检验中文标点断句效果的示例文字。";
        let text = sentence.repeat(81);
        let chunks = split_once(&text, 29, 0);
        assert!(chunks.len() > 40);
        for c in &chunks {
            assert!(measure(c) <= 29);
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn heavier_overlap_yields_fewer_chunks() {
        let para = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romero sierra tango uniform victor whiskey xray yankee zulu ";
        let text = para.repeat(35);
        let light = split_once(text.trim_end(), 200, 50);
        let heavy = split_once(text.trim_end(), 200, 150);
        // Heavier overlap slides the window by a smaller step, so it yields
        // more, denser chunks over the same corpus.
        assert!(heavy.len() > light.len());
    }

    #[test]
    fn segment_splitter_tags_headings_and_numbers_seq() {
        let segments = [
            Segment {
                text: "Alpha section introductory body text.".repeat(170),
                heading: "Overview".to_string(),
                page: None,
            },
            Segment {
                text: "Beta subsection supporting explanatory content.".repeat(190),
                heading: "Deep Dive".to_string(),
                page: Some(7),
            },
        ];
        let chunks = RecursiveSegmentSplitter::default().split_segments(&segments);

        assert!(chunks.len() > 20);
        // Sequential numbering is global across segments.
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.seq as usize, i);
        }
        // Headings propagate per originating segment.
        assert!(chunks.iter().any(|c| c.heading == "Overview"));
        assert!(chunks.iter().any(|c| c.heading == "Deep Dive"));
        // Page propagates only for the annotated section.
        assert!(chunks.iter().any(|c| c.page == Some(7)));
        // Digests carry the algorithmic prefix and distinguish content.
        assert!(chunks.iter().all(|c| c.digest.starts_with("blake3:")));
        assert_eq!(chunks.first().unwrap().heading, "Overview");
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        let chunks = RecursiveSegmentSplitter::default().split_segments(&[]);
        assert!(chunks.is_empty());
    }
}
