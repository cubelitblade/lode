//! Workspace-relative path conventions.
//!
//! Workspace-relative paths are UTF-8 strings carrying posix text: they live
//! in the domain model, the index database, and JSON payloads, and must
//! byte-match across machines. Only the human-facing render layer converts
//! them to OS-native paths.
//!
//! # Contract
//!
//! - Persistence (SQLite `files.path` column) and JSON structured fields
//!   always carry posix text.
//! - Human-facing output (render layer, error prose, progress bars) converts
//!   to OS-native paths via [`WorkspacePath::to_native`].
//! - Never mix path flavours: use `root.join(path.as_str())` (joining with a
//!   string slice), not `root.join(Path::new(path.as_str()))`.
//!
//! # Domain type
//!
//! [`WorkspacePath`] is the domain type for workspace-relative paths. It is a
//! newtype over a posix-text `String`, so the compiler distinguishes it from
//! arbitrary strings and OS-native paths. This mirrors Python's
//! `PurePosixPath` and is reused across the whole codebase (records, store,
//! pipeline, CLI).
//!
//! # Boundary rule
//!
//! Once a path enters the domain layer it must be a [`WorkspacePath`]. Plain
//! `Path`/`PathBuf` (OS-native) and `String` (arbitrary text) are only used
//! at the boundaries — filesystem walks, SQLite rows, JSON payloads, CLI
//! render — and are converted across the boundary via the conversion
//! functions:
//!
//! - `Path` → `WorkspacePath`: [`WorkspacePath::from_native`]
//! - `String` (posix text) → `WorkspacePath`: [`WorkspacePath::from_posix`]
//! - `WorkspacePath` → `PathBuf`: [`WorkspacePath::to_native`]
//! - `WorkspacePath` → `String` (posix text): [`WorkspacePath::as_str`]

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A workspace-relative path: posix text, platform-independent.
///
/// This is the domain type for all workspace-relative paths. It carries
/// posix text (forward slashes) so it byte-matches across machines (e.g.
/// WSL reading a Windows checkout through `/mnt/c`). Convert to OS-native
/// paths only at the render/IO boundary via [`WorkspacePath::to_native`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkspacePath(String);

impl WorkspacePath {
    /// Build from a posix-text string.
    ///
    /// The input must already be posix text (forward slashes). Prefer
    /// [`WorkspacePath::from_native`] when converting from an OS-native
    /// `Path` produced by a filesystem walk.
    pub fn from_posix(posix: impl Into<String>) -> Self {
        Self(posix.into())
    }

    /// Build from an OS-native `Path`, normalizing separators to posix.
    ///
    /// On Unix this is essentially a no-op. On Windows, `\` separators are
    /// converted to `/`.
    pub fn from_native(path: &Path) -> Self {
        Self(path.to_string_lossy().replace('\\', "/"))
    }

    /// The posix-text representation as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The posix-text string.
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Convert to an OS-native `Path` for disk I/O or human display.
    ///
    /// Rebuilds from parts instead of passing the string to `PathBuf::from`,
    /// so no behaviour depends on mixing path flavours: on Windows, `/`
    /// separators in the posix text are converted to the native `\`.
    ///
    /// On Unix this is essentially a no-op.
    pub fn to_native(&self) -> PathBuf {
        let mut buf = PathBuf::new();
        for part in self.0.split('/') {
            buf.push(part);
        }
        buf
    }

    /// Convert to an OS-native string for display.
    ///
    /// Combines [`WorkspacePath::to_native`] with `to_string_lossy`.
    pub fn to_native_display(&self) -> String {
        self.to_native().to_string_lossy().into_owned()
    }
}

impl fmt::Display for WorkspacePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for WorkspacePath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for WorkspacePath {
    fn from(s: &str) -> Self {
        Self::from_posix(s)
    }
}

impl From<String> for WorkspacePath {
    fn from(s: String) -> Self {
        Self::from_posix(s)
    }
}

impl From<&WorkspacePath> for String {
    fn from(p: &WorkspacePath) -> Self {
        p.0.clone()
    }
}

impl Serialize for WorkspacePath {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for WorkspacePath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::from_posix(s))
    }
}

/// Normalize an OS-native path to a workspace-relative posix string.
///
/// Convenience free function for callers that only need the string form
/// (e.g. building a `WorkspacePath` from a walk result). Prefer
/// [`WorkspacePath::from_native`] when the result is a domain value.
pub fn to_rep(path: &Path) -> String {
    WorkspacePath::from_native(path).into_inner()
}

/// Convert a posix-text workspace-relative path to an OS-native `Path`.
///
/// Convenience free function for the render layer. Prefer
/// [`WorkspacePath::to_native`] when the input is already a domain value.
pub fn to_native(rel: &str) -> PathBuf {
    WorkspacePath::from_posix(rel).to_native()
}

/// Convert a posix-text workspace-relative path to a native string for display.
///
/// Convenience free function for the render layer. Prefer
/// [`WorkspacePath::to_native_display`] when the input is already a domain
/// value.
pub fn to_native_display(rel: &str) -> String {
    WorkspacePath::from_posix(rel).to_native_display()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_posix_and_traits() {
        // from_posix, from &str, from String all produce the same value.
        let from_str: WorkspacePath = "a/b".into();
        let from_string: WorkspacePath = String::from("a/b").into();
        let from_posix = WorkspacePath::from_posix("a/b");
        assert_eq!(from_str, from_posix);
        assert_eq!(from_string, from_posix);
        assert_eq!(from_posix.as_str(), "a/b");
        assert_eq!(from_posix.as_ref(), "a/b");
    }

    #[test]
    fn from_native_normalizes_separators() {
        let p = WorkspacePath::from_native(Path::new("src/main.rs"));
        assert_eq!(p.as_str(), "src/main.rs");
    }

    #[test]
    fn to_native_and_display() {
        let p = WorkspacePath::from_posix("docs/readme.md");
        assert_eq!(p.to_native(), PathBuf::from("docs").join("readme.md"));
        assert_eq!(p.to_native_display(), "docs/readme.md");
        assert_eq!(p.to_string(), "docs/readme.md");
    }

    #[test]
    fn to_native_rebuilds_from_parts() {
        // Each `/`-separated segment is pushed individually, so on Windows
        // this yields `\` separators, not `/`.
        let p = WorkspacePath::from_posix("a/b/c");
        assert_eq!(p.to_native(), PathBuf::from("a").join("b").join("c"));
    }

    #[test]
    fn to_native_empty() {
        let p = WorkspacePath::from_posix("").to_native();
        assert_eq!(p, PathBuf::from(""));
    }

    #[test]
    fn equality_and_ordering() {
        assert_eq!(
            WorkspacePath::from_posix("a/b"),
            WorkspacePath::from_posix("a/b")
        );
        assert_ne!(
            WorkspacePath::from_posix("a/b"),
            WorkspacePath::from_posix("a/c")
        );
        let mut paths = vec![
            WorkspacePath::from_posix("c"),
            WorkspacePath::from_posix("a"),
            WorkspacePath::from_posix("b"),
        ];
        paths.sort();
        assert_eq!(
            paths,
            vec![
                WorkspacePath::from_posix("a"),
                WorkspacePath::from_posix("b"),
                WorkspacePath::from_posix("c"),
            ]
        );
    }

    #[test]
    fn serde_roundtrip() {
        let p = WorkspacePath::from_posix("docs/readme.md");
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"docs/readme.md\"");
        let back: WorkspacePath = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn to_rep_from_native_path() {
        assert_eq!(to_rep(Path::new("a/b/c")), "a/b/c");
    }

    #[test]
    fn free_function_to_native() {
        let p = to_native("a/b/c");
        assert_eq!(p, PathBuf::from("a").join("b").join("c"));
        assert_eq!(to_native_display("docs/readme.md"), "docs/readme.md");
    }
}
