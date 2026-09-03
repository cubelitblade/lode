//! Two-stage update pipeline: detect changes + sync.
//!
//! 1a scope: `classify` (pure disk-vs-index stat comparison) and
//! `detect_changes` (classify + mark stale). `sync` is deferred to 1b.
//!
//! The pipeline uses a Rust-native event model: [`Change`] enum variants
//! represent state transitions, and [`DetectResult`] collects them alongside
//! skipped (unsupported) files. This differs from Python's bucket-list
//! `DetectResult` dataclass.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::index::records::{FileRecord, FileStatus};
use crate::index::store::Store;
use crate::ingestion::digest::file_digest;
use crate::ingestion::formats::is_ingestable;
use crate::relpath::WorkspacePath;

/// Disk state of a file (no status — status belongs to the index side).
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub path: WorkspacePath,
    pub digest: String,
    pub mtime: f64,
    pub size: u64,
}

/// A state transition that the index needs to apply, driving `sync`.
#[derive(Debug)]
pub enum Change {
    /// On disk but not indexed — needs ingestion.
    Added(FileSnapshot),
    /// Indexed but content or stat differs — needs re-embedding.
    Modified { old: FileRecord, new: FileSnapshot },
    /// Indexed but gone from disk — needs removal.
    Removed(FileRecord),
    /// Content digest matches a removed path — needs path re-pointing.
    Renamed {
        from: WorkspacePath,
        to: WorkspacePath,
    },
}

/// Result of a detection-only pass over the workspace.
///
/// `changes` is a non-overlapping event list: each file appears in exactly
/// one `Change` variant (or is absent from changes entirely if unchanged).
/// `unchanged` holds paths whose stat matches the index and are fresh.
/// `skipped` holds workspace-relative paths of files that exist on disk
/// but are not ingestable (unsupported format).
#[derive(Debug, Default)]
pub struct DetectResult {
    pub changes: Vec<Change>,
    pub unchanged: Vec<WorkspacePath>,
    pub skipped: Vec<WorkspacePath>,
}

impl DetectResult {
    /// Number of files that need sync work (Added + Modified + Removed + Renamed).
    pub fn pending(&self) -> usize {
        self.changes.len()
    }

    /// Number of new files (not yet indexed).
    pub fn added_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, Change::Added(_)))
            .count()
    }

    /// Number of modified files (stat changed, re-embed needed).
    pub fn modified_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, Change::Modified { .. }))
            .count()
    }

    /// Number of removed files (gone from disk).
    pub fn removed_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, Change::Removed(_)))
            .count()
    }

    /// Number of renamed files (content moved to new path).
    pub fn renamed_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, Change::Renamed { .. }))
            .count()
    }

    /// Number of unchanged files (stat matches and fresh).
    pub fn unchanged_count(&self) -> usize {
        self.unchanged.len()
    }
}

/// Classify the workspace against an index snapshot, with no side effects.
///
/// Pure disk-vs-index stat comparison over a `{path: FileRecord}` snapshot.
/// Returns the [`DetectResult`] with change events and skipped files.
///
/// - `Added` — on disk but not indexed.
/// - `Modified` — stat differs (or residual stale marker).
/// - `Removed` — indexed but gone from disk.
/// - `Renamed` — a new file whose content digest equals a removed file's
///   digest; reported as `Renamed { from, to }` and excluded from both
///   Added and Removed. Pairing is 1:1 and deterministic (paths sorted).
/// - `skipped` — on disk, unsupported format.
pub fn classify(
    indexed: &HashMap<WorkspacePath, FileRecord>,
    root: &Path,
    ignore_files: &[&str],
) -> DetectResult {
    let discovered = crate::ingestion::discover::discover(root, ignore_files);

    let mut result = DetectResult::default();

    for rel in &discovered {
        if !is_ingestable(&root.join(rel.as_str())) {
            result.skipped.push(rel.clone());
            continue;
        }

        let abs = root.join(rel.as_str());
        let stat = match fs::metadata(&abs) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mtime = stat
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let size = stat.len();

        let snapshot = FileSnapshot {
            path: rel.clone(),
            digest: String::new(), // lazy — only computed for rename detection
            mtime,
            size,
        };

        match indexed.get(rel) {
            None => {
                result.changes.push(Change::Added(FileSnapshot {
                    path: rel.clone(),
                    ..snapshot
                }));
            }
            Some(known) => {
                if known.mtime == snapshot.mtime && known.size == snapshot.size {
                    if known.status == FileStatus::Stale {
                        // Residual stale: stat matches but a previous run failed.
                        result.changes.push(Change::Modified {
                            old: known.clone(),
                            new: snapshot,
                        });
                    } else {
                        result.unchanged.push(rel.clone());
                    }
                } else {
                    result.changes.push(Change::Modified {
                        old: known.clone(),
                        new: snapshot,
                    });
                }
            }
        }
    }

    // Removed: indexed but not on disk.
    let on_disk: std::collections::HashSet<&WorkspacePath> = discovered.iter().collect();
    for (path, record) in indexed {
        if !on_disk.contains(path) {
            result.changes.push(Change::Removed(record.clone()));
        }
    }

    // Pair renames: new file whose digest matches a removed file's digest.
    pair_renames(&mut result, root);

    result
}

/// Detect workspace changes and mark changed files stale.
///
/// Thin wrapper over [`classify`]: snapshots the index, classifies, then
/// applies the stale side effects (flipping `files.status` to STALE). No
/// content is embedded or replaced here — that is `sync`'s job.
pub fn detect_changes(
    store: &Store,
    root: &Path,
    ignore_files: &[&str],
) -> crate::Result<DetectResult> {
    let indexed: HashMap<WorkspacePath, FileRecord> = store
        .list_files()?
        .into_iter()
        .map(|r| (r.path.clone(), r))
        .collect();

    let result = classify(&indexed, root, ignore_files);

    // Mark modified and removed files as stale.
    for change in &result.changes {
        match change {
            Change::Modified { old, .. } | Change::Removed(old) => {
                store.mark_stale(&old.path)?;
            }
            _ => {}
        }
    }

    Ok(result)
}

/// Fold exact-content moves out of Added/Removed into Renamed pairs.
///
/// Only Added files are read (one digest each); the Removed side already
/// carries its digest in the index snapshot. An Added file whose digest
/// matches a Removed path is a move: the content is already indexed, so
/// `sync` can re-point it at zero embedding cost.
fn pair_renames(result: &mut DetectResult, root: &Path) {
    // Collect Removed paths by digest for matching.
    let mut removed_by_digest: HashMap<String, Vec<WorkspacePath>> = HashMap::new();
    for change in &result.changes {
        if let Change::Removed(record) = change {
            removed_by_digest
                .entry(record.digest.clone())
                .or_default()
                .push(record.path.clone());
        }
    }
    // Sort for deterministic pairing.
    for paths in removed_by_digest.values_mut() {
        paths.sort();
    }

    let mut paired_removed: Vec<WorkspacePath> = Vec::new();
    let mut renamed_pairs: Vec<(WorkspacePath, WorkspacePath)> = Vec::new();

    // Collect Added paths for iteration.
    let added_paths: Vec<WorkspacePath> = result
        .changes
        .iter()
        .filter_map(|c| match c {
            Change::Added(snap) => Some(snap.path.clone()),
            _ => None,
        })
        .collect();

    for rel in &added_paths {
        let abs = root.join(rel.as_str());
        let data = match fs::read(&abs) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let digest = file_digest(&data);
        if let Some(candidates) = removed_by_digest.get_mut(&digest) {
            if let Some(old) = candidates.first().cloned() {
                candidates.remove(0);
                paired_removed.push(old.clone());
                renamed_pairs.push((old, rel.clone()));
            }
        }
    }

    if !renamed_pairs.is_empty() {
        // Remove paired entries from Added and Removed.
        result
            .changes
            .retain(|c| !matches!(c, Change::Added(snap) if added_paths.contains(&snap.path)));
        result
            .changes
            .retain(|c| !matches!(c, Change::Removed(r) if paired_removed.contains(&r.path)));

        // Add Renamed events.
        for (from, to) in renamed_pairs {
            result.changes.push(Change::Renamed { from, to });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: build a FileRecord from components.
    fn record(path: &str, digest: &str, mtime: f64, size: u64, status: FileStatus) -> FileRecord {
        FileRecord {
            path: WorkspacePath::from_posix(path),
            digest: digest.to_string(),
            mtime,
            size,
            status,
        }
    }

    #[test]
    fn classify_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("new.txt"), "hello").unwrap();

        let indexed = HashMap::new();
        let result = classify(&indexed, root, &[]);
        assert_eq!(result.added_count(), 1);
        assert_eq!(result.modified_count(), 0);
        assert_eq!(result.removed_count(), 0);
    }

    #[test]
    fn classify_unchanged_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("doc.txt"), "content").unwrap();
        let stat = fs::metadata(root.join("doc.txt")).unwrap();
        let mtime = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut indexed = HashMap::new();
        indexed.insert(
            WorkspacePath::from_posix("doc.txt"),
            record(
                "doc.txt",
                "blake3:abc",
                mtime,
                stat.len(),
                FileStatus::Fresh,
            ),
        );

        let result = classify(&indexed, root, &[]);
        assert_eq!(result.changes.len(), 0);
        assert_eq!(result.unchanged_count(), 1);
    }

    #[test]
    fn classify_modified_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("doc.txt"), "new content").unwrap();

        let mut indexed = HashMap::new();
        indexed.insert(
            WorkspacePath::from_posix("doc.txt"),
            record("doc.txt", "blake3:old", 0.0, 100, FileStatus::Fresh),
        );

        let result = classify(&indexed, root, &[]);
        assert_eq!(result.modified_count(), 1);
    }

    #[test]
    fn classify_removed_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let mut indexed = HashMap::new();
        indexed.insert(
            WorkspacePath::from_posix("deleted.txt"),
            record("deleted.txt", "blake3:abc", 0.0, 50, FileStatus::Fresh),
        );

        let result = classify(&indexed, root, &[]);
        assert_eq!(result.removed_count(), 1);
    }

    #[test]
    fn classify_skipped_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("image.png"), "data").unwrap();

        let indexed = HashMap::new();
        let result = classify(&indexed, root, &[]);
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(result.added_count(), 0);
    }

    #[test]
    fn classify_residual_stale() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("doc.txt"), "content").unwrap();
        let stat = fs::metadata(root.join("doc.txt")).unwrap();
        let mtime = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut indexed = HashMap::new();
        indexed.insert(
            WorkspacePath::from_posix("doc.txt"),
            record(
                "doc.txt",
                "blake3:abc",
                mtime,
                stat.len(),
                FileStatus::Stale,
            ),
        );

        let result = classify(&indexed, root, &[]);
        // Stat matches but status is Stale → Modified (retry).
        assert_eq!(result.modified_count(), 1);
    }

    #[test]
    fn detect_result_counts() {
        let mut result = DetectResult::default();
        result.changes.push(Change::Added(FileSnapshot {
            path: WorkspacePath::from_posix("a.txt"),
            digest: "blake3:aaa".into(),
            mtime: 0.0,
            size: 10,
        }));
        result.changes.push(Change::Modified {
            old: record("b.txt", "blake3:bbb", 0.0, 20, FileStatus::Fresh),
            new: FileSnapshot {
                path: WorkspacePath::from_posix("b.txt"),
                digest: "blake3:bbb_new".into(),
                mtime: 1.0,
                size: 25,
            },
        });
        result.changes.push(Change::Removed(record(
            "c.txt",
            "blake3:ccc",
            0.0,
            30,
            FileStatus::Fresh,
        )));

        assert_eq!(result.pending(), 3);
        assert_eq!(result.added_count(), 1);
        assert_eq!(result.modified_count(), 1);
        assert_eq!(result.removed_count(), 1);
    }
}
