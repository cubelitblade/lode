#![warn(clippy::pedantic)]

//! Public `lexical_details` adapter cost, separate from retrieval and I/O.

use std::hint::black_box;
use std::time::Instant;

use lode_core::embeddings::base::Embedder;
use lode_core::index::ranking::{Fusion, RetrievalPlan};
use lode_core::index::records::{FileRecord, FileStatus, Source};
use lode_core::index::search::explain_with_fts;
use lode_core::index::store::Store;
use lode_core::ingestion::digest::chunk_digest;
use lode_core::ingestion::types::Chunk;
use lode_core::relpath::WorkspacePath;

struct NoNetwork;

impl Embedder for NoNetwork {
    fn model_id(&self) -> lode_core::Result<String> {
        Ok("test-model".into())
    }
    fn dimension(&self) -> lode_core::Result<usize> {
        Ok(4)
    }
    fn embed_query(&self, _: &str) -> lode_core::Result<Vec<f32>> {
        panic!("semantic disabled")
    }
    fn embed_documents(&self, _: &[String]) -> lode_core::Result<Vec<Vec<f32>>> {
        panic!("not needed")
    }
}

#[test]
#[ignore = "opt-in release public JSON adapter measurement; no latency thresholds"]
fn public_json_performance() {
    assert!(!black_box(cfg!(debug_assertions)), "Use --release");
    let output =
        std::env::var_os("LODE_FTS_JSON_REPORT").expect("set absolute LODE_FTS_JSON_REPORT");
    let directory = tempfile::tempdir().unwrap();
    let mut measurements = Vec::new();
    for repetitions in [1, 128, 1024] {
        let text = "alpha beta common ".repeat(repetitions);
        let mut store = Store::open(
            &directory.path().join(format!("{repetitions}.db")),
            "test-model",
            4,
            "unicode61",
        )
        .unwrap();
        store
            .replace_file(
                &FileRecord {
                    path: WorkspacePath::from_posix("synthetic.txt"),
                    digest: format!("blake3:serialization-{repetitions}"),
                    mtime: 1.0,
                    size: u64::try_from(text.len()).unwrap(),
                    status: FileStatus::Fresh,
                },
                "text",
                &[Chunk {
                    digest: chunk_digest(&text),
                    text,
                    seq: 0,
                    heading: String::new(),
                    page: None,
                    span: None,
                }],
                None,
            )
            .unwrap();
        let plan = RetrievalPlan {
            norm: None,
            fusion: Fusion::Linear {
                weights: [(Source::Semantic, 0.0), (Source::Lexical, 1.0)].into(),
            },
        };
        let result = explain_with_fts(&store, &NoNetwork, "alpha beta", 1, &plan, 10).unwrap();
        let reference = serde_json::to_vec(&super::details(&result.lexical_details)).unwrap();
        let mut samples = Vec::new();
        for _ in 0..21 {
            let start = Instant::now();
            let payload = black_box(super::details(&result.lexical_details));
            let bytes = black_box(serde_json::to_vec(&payload).unwrap());
            samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
            assert_eq!(bytes, reference);
        }
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        measurements.push(serde_json::json!({
            "token_repetitions": repetitions,
            "target_bytes": result.explanation.chunk.text.len(),
            "hits": result.lexical_details.matched.as_ref().unwrap().phrases.iter().map(|phrase| phrase.hits.len()).sum::<usize>(),
            "json_bytes": reference.len(),
            "samples_us": samples,
            "median_us": sorted[sorted.len()/2],
            "min_us": sorted[0],
            "max_us": sorted[sorted.len()-1],
        }));
    }
    let compiler = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .unwrap();
    let report = serde_json::json!({
        "compiler": String::from_utf8(compiler.stdout).unwrap().trim(),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "release_build": !cfg!(debug_assertions),
        "boundary": "production lexical_json::details plus serde_json::to_vec only; excludes retrieval, mapping, outer why fields, terminal formatting, process and stdout I/O",
        "measurements": measurements,
    });
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
