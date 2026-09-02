//! Workspace-relative path rules.
//!
//! The domain type for workspace-relative paths is a UTF-8 string carrying
//! posix text. Persistence and JSON structured fields always carry posix
//! text; human-facing output converts via `to_native`.
