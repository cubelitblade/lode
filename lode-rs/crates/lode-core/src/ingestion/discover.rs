//! Workspace file discovery + gitignore-style ignore rules.
//!
//! Ported from `src/lode/ingestion/discover.py`. Walks the workspace and
//! returns workspace-relative paths (as [`WorkspacePath`]) that survive the
//! ignore rules.
//!
//! Uses the `ignore` crate for full gitignore pattern semantics (anchored,
//! unanchored, negation, `**`, `?`, character classes). The walk is powered
//! by [`ignore::WalkBuilder`], which also handles `.gitignore` files when
//! present.

use std::fs;
use std::path::Path;

use ignore::gitignore::Gitignore;

use crate::relpath::WorkspacePath;

/// Always ignored: the runtime data directory holding the index store.
pub const DEFAULT_IGNORES: &[&str] = &[".lode/**"];

/// First-class ignore file, always loaded when present at the workspace root.
pub const LODEIGNORE: &str = ".lodeignore";

/// Discover workspace files, applying ignore rules.
///
/// Returns workspace-relative paths (posix text) of all files under `root`
/// that survive the ignore rules. Mirrors Python's `discover()`.
///
/// Ignore sources (in priority order):
/// 1. Built-in defaults (`.lode/**`)
/// 2. `.lodeignore` at workspace root (always loaded when present)
/// 3. Additional files listed in `ignore_files` (read relative to root)
///
/// Ignore files themselves are excluded from the result.
pub fn discover(root: &Path, ignore_files: &[&str]) -> Vec<WorkspacePath> {
    let gi = build_ignore_spec(root, ignore_files);
    let excluded = ignore_file_paths(ignore_files);

    let mut files = Vec::new();
    for entry in ignore::WalkBuilder::new(root)
        // Don't skip hidden files — the ignore spec handles `.lode/**`.
        .hidden(false)
        // Don't respect the parent's .gitignore — we build our own.
        .git_ignore(false)
        .build()
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        // Skip directories (walk emits them but we only want files).
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        // Skip the root itself (WalkBuilder emits it for the root dir).
        if path == root {
            continue;
        }

        // Build the workspace-relative posix path.
        let rel = match path.strip_prefix(root) {
            Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };

        // Ignore files are never returned.
        if excluded.contains(&rel) {
            continue;
        }

        // Apply the composed ignore spec (DEFAULT_IGNORES + lodeignore + config).
        // `matched_path_or_any_parents` handles both anchored and unanchored
        // gitignore semantics correctly.
        if gi.matched_path_or_any_parents(&rel, false).is_ignore() {
            continue;
        }

        files.push(WorkspacePath::from_posix(rel));
    }

    files.sort();
    files
}

/// Build a combined gitignore spec from defaults + `.lodeignore` + config files.
///
/// The `ignore` crate's [`Gitignore`] handles the full gitignore pattern
/// language: anchored vs unanchored, negation, `**`, `?`, character classes,
/// and directory-only (`dir/`) patterns.
fn build_ignore_spec(root: &Path, ignore_files: &[&str]) -> Gitignore {
    let mut builder = ignore::gitignore::GitignoreBuilder::new(root);

    for line in DEFAULT_IGNORES {
        let _ = builder.add_line(None, line);
    }

    let lodeignore = root.join(LODEIGNORE);
    if lodeignore.is_file()
        && let Ok(text) = fs::read_to_string(&lodeignore)
    {
        for line in text.lines() {
            let _ = builder.add_line(Some(lodeignore.clone()), line);
        }
    }

    for name in ignore_files {
        let path = root.join(name);
        if path.is_file()
            && let Ok(text) = fs::read_to_string(&path)
        {
            for line in text.lines() {
                let _ = builder.add_line(Some(path.clone()), line);
            }
        }
    }

    builder.build().unwrap_or_else(|_| Gitignore::empty())
}

/// Workspace-relative paths of the ignore files, so they never get indexed.
fn ignore_file_paths(ignore_files: &[&str]) -> Vec<String> {
    let mut paths = vec![LODEIGNORE.to_string()];
    paths.extend(ignore_files.iter().map(|s| s.to_string()));
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temp workspace and run discover on it.
    fn discover_in(dir: &Path, ignore_files: &[&str]) -> Vec<String> {
        discover(dir, ignore_files)
            .into_iter()
            .map(|wp| wp.into_inner())
            .collect()
    }

    #[test]
    fn default_ignores_lode_dir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.py"), "x").unwrap();
        fs::create_dir_all(root.join(".lode")).unwrap();
        fs::write(root.join(".lode/index.db"), "db").unwrap();

        let files = discover_in(root, &[]);
        assert!(files.contains(&"src/main.py".to_string()));
        assert!(!files.iter().any(|f| f.starts_with(".lode/")));
    }

    #[test]
    fn lodeignore_basic() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.txt"), "a").unwrap();
        fs::write(root.join("b.txt"), "b").unwrap();
        fs::write(root.join(".lodeignore"), "*.tmp\n").unwrap();
        fs::write(root.join("c.tmp"), "c").unwrap();

        let files = discover_in(root, &[]);
        assert!(files.contains(&"a.txt".to_string()));
        assert!(files.contains(&"b.txt".to_string()));
        assert!(!files.contains(&"c.tmp".to_string()));
    }

    #[test]
    fn lodeignore_dir_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("file.txt"), "x").unwrap();
        fs::create_dir_all(root.join("build")).unwrap();
        fs::write(root.join("build/out.txt"), "y").unwrap();
        fs::write(root.join(".lodeignore"), "build/\n").unwrap();

        let files = discover_in(root, &[]);
        assert!(files.contains(&"file.txt".to_string()));
        assert!(!files.iter().any(|f| f.starts_with("build/")));
    }

    #[test]
    fn lodeignore_negation() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("keep.tmp"), "k").unwrap();
        fs::write(root.join("drop.tmp"), "d").unwrap();
        fs::write(root.join(".lodeignore"), "*.tmp\n!keep.tmp\n").unwrap();

        let files = discover_in(root, &[]);
        assert!(files.contains(&"keep.tmp".to_string()));
        assert!(!files.contains(&"drop.tmp".to_string()));
    }

    #[test]
    fn custom_ignore_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.py"), "a").unwrap();
        fs::write(root.join("b.py"), "b").unwrap();
        fs::write(root.join("custom.ignore"), "*.py\n").unwrap();

        let files = discover_in(root, &["custom.ignore"]);
        assert!(files.is_empty());
    }

    #[test]
    fn nested_ignored_dir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("top.txt"), "t").unwrap();
        fs::create_dir_all(root.join("a/b/c")).unwrap();
        fs::write(root.join("a/b/c/deep.txt"), "d").unwrap();
        fs::write(root.join(".lodeignore"), "a/b/\n").unwrap();

        let files = discover_in(root, &[]);
        assert!(files.contains(&"top.txt".to_string()));
        assert!(!files.iter().any(|f| f.starts_with("a/b/")));
    }

    #[test]
    fn sorted_output() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("z.txt"), "z").unwrap();
        fs::write(root.join("a.txt"), "a").unwrap();
        fs::write(root.join("m.txt"), "m").unwrap();

        let files = discover_in(root, &[]);
        assert_eq!(files, vec!["a.txt", "m.txt", "z.txt"]);
    }

    #[test]
    fn ignore_files_excluded_from_result() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("content.txt"), "c").unwrap();
        fs::write(root.join(".lodeignore"), "# patterns\n").unwrap();
        fs::write(root.join("extra.ignore"), "*.txt\n").unwrap();

        let files = discover_in(root, &["extra.ignore"]);
        // .lodeignore and extra.ignore are excluded; content.txt is filtered by extra.ignore
        assert!(files.is_empty());
    }
}
