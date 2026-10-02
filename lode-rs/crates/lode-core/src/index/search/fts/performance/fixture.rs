#![warn(clippy::pedantic)]

use std::path::Path;

use crate::embeddings::testing::file_record;
use crate::index::store::Store;
use crate::ingestion::digest::chunk_digest;
use crate::ingestion::types::Chunk;

pub(super) const TARGET_TEXT: &str = "alpha beta common alphabet 知识 中国 重庆 needleabc";

pub(super) fn build(path: &Path, strategy: &str, rows: u32) -> crate::Result<Store> {
    let mut store = Store::open(path, "test-model", 4, strategy)?;
    let chunks: Vec<_> = (0..rows)
        .map(|seq| {
            let text = if seq == 0 {
                TARGET_TEXT.to_owned()
            } else if seq == 1 {
                format!(
                    "{} needleabc",
                    "alpha beta common alphabet 知识 中国 重庆 ".repeat(512)
                )
            } else {
                format!("alpha beta common alphabet 知识 中国 filler{seq}")
            };
            Chunk {
                digest: chunk_digest(&text),
                text,
                seq,
                heading: String::new(),
                page: None,
                span: None,
            }
        })
        .collect();
    // Fixed small vectors isolate FTS costs from embedding/network variance.
    let vectors = vec![vec![0.1, 0.2, 0.3, 0.4]; chunks.len()];
    store.replace_file(
        &file_record(
            "synthetic.txt",
            "blake3:performance-fixture",
            u64::from(rows),
        ),
        "text",
        &chunks,
        Some(&vectors),
    )?;
    Ok(store)
}
