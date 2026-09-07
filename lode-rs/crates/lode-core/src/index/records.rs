//! Data records shared across the index layer: enums, row types.
//!
//! Mirrors `src/lode/index/store/records.py` for schema compatibility.

use crate::relpath::WorkspacePath;
use serde::{Deserialize, Serialize};

/// Result of comparing the stored model with the current embedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    /// Stored model matches the current embedder.
    Match,
    /// Stored model differs from the current embedder.
    Mismatch,
    /// Embedder unreachable; the store stays usable for cached queries.
    Unknown,
}

/// Freshness status of an indexed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileStatus {
    /// Content is up to date.
    Fresh,
    /// Content has changed on disk; re-embedding is pending.
    Stale,
}

impl FileStatus {
    /// The string representation stored in SQLite.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Stale => "stale",
        }
    }
}

impl std::str::FromStr for FileStatus {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fresh" => Ok(Self::Fresh),
            "stale" => Ok(Self::Stale),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for FileStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One indexed path joined with its content metadata (mirrors `files` ⋈ `contents`).
///
/// `path` is the workspace-relative domain type (see [`crate::relpath`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileRecord {
    /// Workspace-relative posix path.
    pub path: WorkspacePath,
    /// Content digest: `blake3:<hex>`.
    pub digest: String,
    /// File modification time (epoch seconds).
    pub mtime: f64,
    /// File size in bytes.
    pub size: u64,
    /// Freshness status.
    pub status: FileStatus,
}

/// One workspace path referencing a shared content, with its freshness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathRef {
    /// Workspace-relative posix path.
    pub path: WorkspacePath,
    /// Freshness status.
    pub status: FileStatus,
}

/// A retrieval source in the hybrid pipeline.
///
/// Python keys source tables by the string name ("semantic"/"lexical");
/// this is the closed-set equivalent so match sites are exhaustive and JSON
/// keys stay identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Dense vectors via the vec0 table; cosine similarity scores.
    Semantic,
    /// FTS5 via the chunks_fts table; BM25 scores.
    Lexical,
}

impl Source {
    /// The source name used in Python tables and JSON payloads.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Lexical => "lexical",
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Source {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "semantic" => Ok(Self::Semantic),
            "lexical" => Ok(Self::Lexical),
            _ => Err(()),
        }
    }
}

/// One indexed chunk joined with every path referencing its content.
///
/// Mirrors Python's `ChunkWithPath`: content is shared by identical files, so
/// a chunk carries every `files` row pointing at its content. Used by search
/// result assembly and by `dig`'s digest lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkWithRefs {
    /// Content digest: `blake3:<hex>`.
    pub digest: String,
    /// Chunk body text.
    pub text: String,
    /// Heading chain the chunk belongs to (empty when top-level).
    pub heading: String,
    /// Chunk sequence within its content (0-based), `None` only for rows the
    /// caller did not select `seq`.
    pub seq: Option<u32>,
    /// 1-based page number for PDF chunks, `None` for other formats.
    pub page: Option<u32>,
    /// Every workspace path referencing this content, with freshness.
    pub refs: Vec<PathRef>,
}

impl ChunkWithRefs {
    /// Representative reference: smallest fresh path, else smallest overall.
    #[must_use]
    pub fn primary(&self) -> PathRef {
        let mut fresh: Vec<&PathRef> = self
            .refs
            .iter()
            .filter(|r| r.status == FileStatus::Fresh)
            .collect();
        if fresh.is_empty() {
            fresh = self.refs.iter().collect();
        }
        // `expect` is safe: a join-derived chunk always has ≥1 reference.
        fresh
            .into_iter()
            .min_by(|a, b| a.path.as_str().cmp(b.path.as_str()))
            .expect("refs is never empty for a join-derived chunk")
            .clone()
    }
}

/// One kNN hit: chunk rowid with its L2 distance, ordered nearest first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMatch {
    /// SQLite rowid of the chunk.
    pub rowid: i64,
    /// L2 distance (smaller = closer).
    pub distance: f64,
}

/// One BM25 hit: chunk rowid with its score, ordered best first.
///
/// SQLite BM25 scores are negative and closer to zero means better, so
/// best-first is descending score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseMatch {
    /// SQLite rowid of the chunk.
    pub rowid: i64,
    /// BM25 score (higher = better).
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_status_display() {
        assert_eq!(FileStatus::Fresh.to_string(), "fresh");
        assert_eq!(FileStatus::Stale.to_string(), "stale");
    }

    #[test]
    fn file_status_roundtrip() {
        let fresh = FileStatus::Fresh;
        let json = serde_json::to_string(&fresh).unwrap();
        assert_eq!(json, "\"fresh\"");
        let back: FileStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fresh);
    }

    #[test]
    fn model_status_roundtrip() {
        for status in [
            ModelStatus::Match,
            ModelStatus::Mismatch,
            ModelStatus::Unknown,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ModelStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn file_record_construction() {
        let record = FileRecord {
            path: WorkspacePath::from_posix("docs/readme.md"),
            digest: "blake3:abcdef".into(),
            mtime: 1234567890.0,
            size: 1024,
            status: FileStatus::Fresh,
        };
        assert_eq!(record.path.as_str(), "docs/readme.md");
        assert_eq!(record.status, FileStatus::Fresh);
    }

    #[test]
    fn dense_match_ordering() {
        let mut matches = [
            DenseMatch {
                rowid: 1,
                distance: 0.5,
            },
            DenseMatch {
                rowid: 2,
                distance: 0.1,
            },
            DenseMatch {
                rowid: 3,
                distance: 0.3,
            },
        ];
        matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        assert_eq!(matches[0].rowid, 2);
        assert_eq!(matches[1].rowid, 3);
        assert_eq!(matches[2].rowid, 1);
    }

    #[test]
    fn sparse_match_ordering() {
        let mut matches = [
            SparseMatch {
                rowid: 1,
                score: -5.0,
            },
            SparseMatch {
                rowid: 2,
                score: -1.0,
            },
            SparseMatch {
                rowid: 3,
                score: -3.0,
            },
        ];
        // BM25: higher (closer to zero) is better → descending.
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(matches[0].rowid, 2);
        assert_eq!(matches[1].rowid, 3);
        assert_eq!(matches[2].rowid, 1);
    }
}
