//! lode core library.
//!
//! The single business-logic crate. Both `lode-cli` and `lode-mcp` are thin
//! adapters over this crate.

pub mod config;
pub mod errors;
pub mod relpath;

pub mod embeddings;
pub mod fts;
pub mod index;
pub mod ingestion;
pub mod messages;

pub use errors::{Error, Result};

/// Run a test that initializes the process-global Jieba dictionary with an
/// isolated cache that the parent test process removes after the child exits.
#[cfg(test)]
pub(crate) fn run_with_isolated_jieba_cache(test_name: &str, test: impl FnOnce()) {
    const CHILD_MARKER: &str = "LODE_CORE_JIEBA_TEST_CHILD";
    if std::env::var_os(CHILD_MARKER).is_some() {
        test();
        return;
    }

    let cache = tempfile::tempdir().expect("create isolated Jieba test cache");
    let output = std::process::Command::new(
        std::env::current_exe().expect("resolve current test executable"),
    )
    .args(["--exact", test_name, "--nocapture"])
    .env(CHILD_MARKER, "1")
    .env("LODE_CACHE_DIR", cache.path())
    .output()
    .expect("run Jieba test in subprocess");

    assert!(
        output.status.success(),
        "isolated Jieba test {test_name} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
