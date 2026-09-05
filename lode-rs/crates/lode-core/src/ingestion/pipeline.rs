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

/// Per-file failure recorded during sync.
#[derive(Debug, Clone)]
pub struct FailedFile {
    pub path: WorkspacePath,
    pub error: String,
}

/// Summary of a sync pass over the workspace.
#[derive(Debug, Default)]
pub struct SyncSummary {
    pub added: Vec<WorkspacePath>,
    pub updated: Vec<WorkspacePath>,
    pub removed: Vec<WorkspacePath>,
    pub renamed: Vec<(WorkspacePath, WorkspacePath)>,
    pub failed: Vec<FailedFile>,
    pub unchanged: usize,
    pub skipped: usize,
}

/// Classify the workspace against an index snapshot.
///
/// Disk-vs-index stat comparison over a `{path: FileRecord}` snapshot,
/// followed by rename pairing. Returns the [`DetectResult`] with change
/// events and skipped files.
///
/// Rename pairing reads each Added file once to compute its content digest
/// (the Removed side already carries its digest in the snapshot), so this
/// is not a pure stat pass — but it never writes anything.
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
    store: &mut Store,
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

/// Update the index to match the workspace, consuming a detection result.
///
/// Never classifies: it only works the buckets `detect` computed —
/// `Renamed` (re-pointed at zero embedding cost), `Added` (new files),
/// `Modified` (stat changed, re-extract needed), and `Removed` (gone).
///
/// The 1b scope extracts and chunks text but does not embed vectors;
/// the vec0 table stays empty until 1c lands the embedder.
///
/// Individual file failures are recorded on the summary and never abort
/// the run. `report` (optional) is called as `report(done, total, path)`
/// before each file is processed, letting the CLI surface progress.
//
// The closure signature is the CLI progress contract; factoring it into a
// named alias would add a public type for a single call site.
#[allow(clippy::type_complexity)]
pub fn sync(
    store: &mut Store,
    root: &Path,
    splitter: &crate::ingestion::split::RecursiveSegmentSplitter,
    detect: &DetectResult,
    report: Option<&dyn Fn(usize, usize, Option<&WorkspacePath>)>,
) -> crate::Result<SyncSummary> {
    use crate::ingestion::extract::extract_document;
    use crate::ingestion::split::SegmentSplitter;

    let mut summary = SyncSummary {
        unchanged: detect.unchanged.len(),
        skipped: detect.skipped.len(),
        ..Default::default()
    };

    // Collect paths that need extraction + chunking (adds, modifies, rename fallbacks).
    let mut to_embed: Vec<WorkspacePath> = Vec::new();
    let mut added_set: std::collections::HashSet<WorkspacePath> = std::collections::HashSet::new();
    let mut rename_fallbacks: Vec<WorkspacePath> = Vec::new();
    let mut fallback_removals: Vec<WorkspacePath> = Vec::new();

    // 1. Handle renames: reference + remove old.
    for change in &detect.changes {
        if let Change::Renamed { from, to } = change {
            let abs = root.join(to.as_str());
            let data = match std::fs::read(&abs) {
                Ok(d) => d,
                Err(exc) => {
                    summary.failed.push(FailedFile {
                        path: to.clone(),
                        error: format!("could not read file: {exc}"),
                    });
                    store.remove_file(from)?;
                    continue;
                }
            };
            let stat = match std::fs::metadata(&abs) {
                Ok(s) => s,
                Err(exc) => {
                    summary.failed.push(FailedFile {
                        path: to.clone(),
                        error: format!("could not stat file: {exc}"),
                    });
                    store.remove_file(from)?;
                    continue;
                }
            };
            let mtime = stat
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let record = FileRecord {
                path: to.clone(),
                digest: file_digest(&data),
                mtime,
                size: stat.len(),
                status: FileStatus::Fresh,
            };
            if store.reference_file(&record)? {
                store.remove_file(from)?;
                summary.renamed.push((from.clone(), to.clone()));
            } else {
                // Content changed under us — full re-embed for new path.
                rename_fallbacks.push(to.clone());
                fallback_removals.push(from.clone());
            }
        }
    }

    // 2. Collect files to embed: new files first, then rename fallbacks
    //    (semantically additions), then modified files — matching Python's
    //    `[*new_files, *rename_fallbacks, *changed_files]` reading order.
    //    Rename fallbacks count as additions (Python's `added_paths` includes
    //    them), so they join `added_set` too.
    for change in &detect.changes {
        if let Change::Added(snap) = change {
            added_set.insert(snap.path.clone());
            to_embed.push(snap.path.clone());
        }
    }
    added_set.extend(rename_fallbacks.iter().cloned());
    to_embed.extend(rename_fallbacks);
    for change in &detect.changes {
        if let Change::Modified { new, .. } = change {
            to_embed.push(new.path.clone());
        }
    }

    let total = to_embed.len();
    for (idx, rel) in to_embed.iter().enumerate() {
        if let Some(report) = report {
            report(idx, total, Some(rel));
        }

        let abs = root.join(rel.as_str());
        let data = match std::fs::read(&abs) {
            Ok(d) => d,
            Err(exc) => {
                summary.failed.push(FailedFile {
                    path: rel.clone(),
                    error: format!("could not read file: {exc}"),
                });
                store.mark_stale(rel)?;
                continue;
            }
        };
        let stat = match std::fs::metadata(&abs) {
            Ok(s) => s,
            Err(exc) => {
                summary.failed.push(FailedFile {
                    path: rel.clone(),
                    error: format!("could not stat file: {exc}"),
                });
                store.mark_stale(rel)?;
                continue;
            }
        };
        let mtime = stat
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let digest = file_digest(&data);
        let record = FileRecord {
            path: rel.clone(),
            digest: digest.clone(),
            mtime,
            size: stat.len(),
            status: FileStatus::Fresh,
        };

        // Content-reuse check.
        if store.reference_file(&record)? {
            if added_set.contains(rel) {
                summary.added.push(rel.clone());
            } else {
                summary.updated.push(rel.clone());
            }
            continue;
        }

        // Extract.
        let suffix = rel
            .as_str()
            .rsplit('.')
            .next()
            .map(|s| format!(".{s}"))
            .unwrap_or_default();
        let segments = match extract_document(&data, &suffix) {
            Some(s) => s,
            None => {
                summary.skipped += 1;
                continue;
            }
        };

        // Split + store (no embedding in 1b).
        let chunks = splitter.split_segments(&segments);

        match store.replace_file(&record, &chunks) {
            Ok(_) => {
                if added_set.contains(rel) {
                    summary.added.push(rel.clone());
                } else {
                    summary.updated.push(rel.clone());
                }
            }
            Err(exc) => {
                summary.failed.push(FailedFile {
                    path: rel.clone(),
                    error: exc.to_string(),
                });
                store.mark_stale(rel)?;
            }
        }
    }

    // 3. Remove missing files and rename fallback removals.
    for rel in detect.changes.iter().filter_map(|c| {
        if let Change::Removed(r) = c {
            Some(&r.path)
        } else {
            None
        }
    }) {
        store.remove_file(rel)?;
        summary.removed.push(rel.clone());
    }
    for rel in &fallback_removals {
        store.remove_file(rel)?;
        summary.removed.push(rel.clone());
    }

    if let Some(report) = report {
        report(total, total, None);
    }

    Ok(summary)
}

/// Fold exact-content moves out of Added/Removed into Renamed pairs.
///
/// Only Added files are read (one digest each); the Removed side already
/// carries its digest in the index snapshot. An Added file whose digest
/// matches a Removed path is a move: the content is already indexed, so
/// `sync` can re-point it at zero embedding cost.
///
/// Returns early when there is nothing to pair (no Removed or no Added),
/// so a first `mine` over an empty snapshot never reads file contents.
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
    // Nothing to pair: no removed files (e.g. a first mine over an empty
    // snapshot) — skip reading any Added file contents.
    if removed_by_digest.is_empty() {
        return;
    }
    // Sort for deterministic pairing.
    for paths in removed_by_digest.values_mut() {
        paths.sort();
    }

    // Collect Added paths for iteration, sorted for deterministic pairing
    // (mirrors Python's `sorted(new_files)`).
    let mut added_paths: Vec<WorkspacePath> = result
        .changes
        .iter()
        .filter_map(|c| match c {
            Change::Added(snap) => Some(snap.path.clone()),
            _ => None,
        })
        .collect();
    added_paths.sort();
    if added_paths.is_empty() {
        return;
    }

    let mut paired_added: std::collections::HashSet<WorkspacePath> =
        std::collections::HashSet::new();
    let mut paired_removed: std::collections::HashSet<WorkspacePath> =
        std::collections::HashSet::new();
    let mut renamed_pairs: Vec<(WorkspacePath, WorkspacePath)> = Vec::new();

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
                paired_added.insert(rel.clone());
                paired_removed.insert(old.clone());
                renamed_pairs.push((old, rel.clone()));
            }
        }
    }

    if !renamed_pairs.is_empty() {
        // Remove only the paired entries from Added and Removed; unpaired
        // Added files stay in their bucket.
        result
            .changes
            .retain(|c| !matches!(c, Change::Added(snap) if paired_added.contains(&snap.path)));
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

    #[test]
    fn sync_basic() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("hello.txt"), "hello world").unwrap();

        let db = root.join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();
        let splitter = crate::ingestion::split::RecursiveSegmentSplitter::new(200, 50).unwrap();

        let detect = detect_changes(&mut store, root, &[]).unwrap();
        assert_eq!(detect.added_count(), 1);

        let summary = sync(&mut store, root, &splitter, &detect, None).unwrap();
        assert_eq!(summary.added.len(), 1);
        assert_eq!(summary.failed.len(), 0);

        // File is now indexed.
        let files = store.list_files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path.as_str(), "hello.txt");
    }

    #[test]
    fn sync_removes_deleted_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.txt"), "aaa").unwrap();

        let db = root.join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();
        let splitter = crate::ingestion::split::RecursiveSegmentSplitter::new(200, 50).unwrap();

        // First sync: index the file.
        let detect = detect_changes(&mut store, root, &[]).unwrap();
        sync(&mut store, root, &splitter, &detect, None).unwrap();
        assert_eq!(store.list_files().unwrap().len(), 1);

        // Delete the file.
        fs::remove_file(root.join("a.txt")).unwrap();
        let detect = detect_changes(&mut store, root, &[]).unwrap();
        assert_eq!(detect.removed_count(), 1);

        let summary = sync(&mut store, root, &splitter, &detect, None).unwrap();
        assert_eq!(summary.removed.len(), 1);
        assert!(store.list_files().unwrap().is_empty());
    }

    #[test]
    fn sync_skips_unsupported_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("image.png"), b"\x89PNG").unwrap();

        // Keep the database outside the walked root so it does not pollute
        // the discovery pass.
        let db_dir = tempfile::tempdir().unwrap();
        let db = db_dir.path().join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();
        let splitter = crate::ingestion::split::RecursiveSegmentSplitter::new(200, 50).unwrap();

        let detect = detect_changes(&mut store, root, &[]).unwrap();
        assert_eq!(detect.skipped.len(), 1);

        let summary = sync(&mut store, root, &splitter, &detect, None).unwrap();
        assert_eq!(summary.skipped, 1);
        assert!(store.list_files().unwrap().is_empty());
    }

    #[test]
    fn classify_pairs_only_matching_renames() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Two new files; only one matches the removed file's digest.
        let moved_content = "same content";
        fs::write(root.join("moved.txt"), moved_content).unwrap();
        fs::write(root.join("brand_new.txt"), "brand new").unwrap();

        let moved_digest = file_digest(moved_content.as_bytes());
        let mut indexed = HashMap::new();
        indexed.insert(
            WorkspacePath::from_posix("old.txt"),
            record("old.txt", &moved_digest, 0.0, 50, FileStatus::Fresh),
        );

        let result = classify(&indexed, root, &[]);
        // moved.txt pairs with old.txt → Renamed; brand_new.txt stays Added.
        assert_eq!(result.renamed_count(), 1);
        assert_eq!(result.added_count(), 1);
        assert_eq!(result.removed_count(), 0);
    }

    #[test]
    fn classify_pairs_are_one_to_one() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Two new files with identical content; only one removed file.
        let content = "duplicate content";
        fs::write(root.join("dup1.txt"), content).unwrap();
        fs::write(root.join("dup2.txt"), content).unwrap();

        let digest = file_digest(content.as_bytes());
        let mut indexed = HashMap::new();
        indexed.insert(
            WorkspacePath::from_posix("old.txt"),
            record("old.txt", &digest, 0.0, 50, FileStatus::Fresh),
        );

        let result = classify(&indexed, root, &[]);
        // Only one duplicate pairs with old.txt; the other stays Added.
        assert_eq!(result.renamed_count(), 1);
        assert_eq!(result.added_count(), 1);
        assert_eq!(result.removed_count(), 0);
    }

    #[test]
    fn sync_rename_fallback_counts_as_added() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.txt"), "same content").unwrap();

        let db = root.join("index.db");
        let mut store = Store::open(&db, 128, "unicode61").unwrap();
        let splitter = crate::ingestion::split::RecursiveSegmentSplitter::new(200, 50).unwrap();

        // Index a.txt.
        let detect = detect_changes(&mut store, root, &[]).unwrap();
        sync(&mut store, root, &splitter, &detect, None).unwrap();

        // Rename a.txt -> b.txt with identical content, then change b.txt's
        // content so the rename pairing succeeds at detect time but the
        // content-reuse check fails at sync time → rename fallback.
        fs::rename(root.join("a.txt"), root.join("b.txt")).unwrap();
        let detect = detect_changes(&mut store, root, &[]).unwrap();
        assert_eq!(detect.renamed_count(), 1);

        fs::write(root.join("b.txt"), "changed content").unwrap();
        let summary = sync(&mut store, root, &splitter, &detect, None).unwrap();
        // The fallback re-embeds b.txt and counts it as added (Python
        // semantics: `added_paths` includes rename fallbacks).
        assert_eq!(summary.added.len(), 1);
        assert_eq!(summary.updated.len(), 0);
        assert_eq!(summary.renamed.len(), 0);
    }
}
