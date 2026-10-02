#![warn(clippy::pedantic)]

//! Opt-in release measurements of production functions, never CI thresholds.

mod fixture;

use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use super::*;
use crate::embeddings::testing::FakeEmbedder;
use crate::fts::explanation::FtsHitMapping;
use crate::index::explanation::ScoreExplanation;
use crate::index::ranking::{Fusion, Norm};

const TEST_NAME: &str = "index::search::fts::performance::production_performance";
const CHILD_ENV: &str = "LODE_FTS_PERFORMANCE_CHILD";
const WARM_SAMPLES: usize = 9;
const REOPEN_SAMPLES: usize = 3;
const TOP_K: u32 = 10;

#[derive(Serialize)]
struct Distribution {
    samples: Vec<f64>,
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

impl Distribution {
    fn new(samples: Vec<f64>) -> Self {
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        Self {
            median_us: sorted[sorted.len() / 2],
            min_us: sorted[0],
            max_us: sorted[sorted.len() - 1],
            samples,
        }
    }
}

#[derive(Serialize)]
struct Measurement {
    strategy: String,
    corpus_rows: u32,
    storage: String,
    cache_condition: String,
    query: String,
    match_expression: String,
    target_rowid: i64,
    target_bytes: usize,
    phrase_count: usize,
    hit_count: usize,
    in_candidate_pool: bool,
    baseline: Distribution,
    diagnostics_without_mapping: Distribution,
    complete: Distribution,
    mapping_only: Distribution,
    additional_diagnostics_median_us: f64,
    additional_complete_median_us: f64,
    complete_minus_diagnostics_median_us: f64,
    max_bm25_error: f64,
}

// Compose the same production primitives as explain_with_fts, omitting only
// map_fts_hits and display metadata, so the mapping boundary is measurable.
fn without_mapping(
    store: &Store,
    embedder: &FakeEmbedder,
    query: &str,
    rowid: i64,
    plan: &RetrievalPlan,
) -> crate::Result<(ScoreExplanation, FtsMatchDetails)> {
    store.with_read_snapshot(|store| {
        let chunk = store.get_chunks(&[rowid])?.remove(&rowid).unwrap();
        let expression =
            store.resolve_fts_query(&crate::fts::match_query(query, store.query_tokenizer())?)?;
        let candidates =
            score_candidates_for_query(store, embedder, query, plan, TOP_K, Some(&expression))?;
        let explanation = explanation_from_candidates(chunk, &candidates, rowid, plan, TOP_K);
        let matched = store
            .explain_fts(&MatchExpr::Prebuilt(expression), rowid)?
            .matched
            .unwrap();
        validate_pool_score(explanation.sources.get(&Source::Lexical), Some(&matched))?;
        Ok((explanation, matched))
    })
}

fn timed<T>(call: impl FnOnce() -> crate::Result<T>) -> crate::Result<(f64, T)> {
    let started = Instant::now();
    let result = black_box(call()?);
    Ok((started.elapsed().as_secs_f64() * 1_000_000.0, result))
}

fn reopen(
    path: &Path,
    strategy: &str,
    original: &Store,
    reopened: bool,
) -> crate::Result<Option<Store>> {
    if reopened {
        Ok(Some(Store::open_existing_with_tokenizer(path, strategy)?))
    } else {
        black_box(original);
        Ok(None)
    }
}

fn validate_surfaces(text: &str, hits: &[&crate::fts::explanation::FtsHit]) {
    for hit in hits {
        let FtsHitMapping::Mapped(span) = &hit.mapping else {
            panic!("valid fixture mapping unavailable: {:?}", hit.mapping);
        };
        assert_eq!(
            text.get(span.start_byte..span.end_byte),
            Some(span.surface.as_str())
        );
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Benchmark matrix dimensions are recorded explicitly"
)]
fn measure(
    original: &Store,
    path: &Path,
    strategy: &str,
    rows: u32,
    storage: &str,
    reopened: bool,
    query: &str,
    rowid: i64,
) -> crate::Result<Measurement> {
    let plan = RetrievalPlan {
        norm: Some(Norm::MinMax),
        fusion: Fusion::Linear {
            weights: [(Source::Semantic, 0.6), (Source::Lexical, 0.4)].into(),
        },
    };
    let embedder = FakeEmbedder::default();
    let expected = explain_with_fts(original, &embedder, query, rowid, &plan, TOP_K)?;
    let raw = without_mapping(original, &embedder, query, rowid, &plan)?;
    assert_eq!(raw.0, expected.explanation);
    assert_eq!(
        super::super::explain(original, &embedder, query, rowid, &plan, TOP_K)?,
        expected.explanation
    );
    let text = &expected.explanation.chunk.text;
    let matched = expected.lexical_details.matched.as_ref().unwrap();
    let hits: Vec<_> = matched
        .phrases
        .iter()
        .flat_map(|phrase| &phrase.hits)
        .collect();
    validate_surfaces(text, &hits);
    let mut baseline = Vec::new();
    let mut diagnostics = Vec::new();
    let mut complete = Vec::new();
    let mut mapping = Vec::new();
    let repetitions = if reopened {
        REOPEN_SAMPLES
    } else {
        WARM_SAMPLES
    };
    let mut max_error = 0.0_f64;
    let before_queries = embedder.query_calls.get();
    for repetition in 0..repetitions {
        // Rotate execution order; each reopened pipeline gets its own connection.
        for phase in 0..3 {
            let opened = reopen(path, strategy, original, reopened)?;
            let store = opened.as_ref().unwrap_or(original);
            match (phase + repetition) % 3 {
                0 => {
                    let (elapsed, result) = timed(|| {
                        super::super::explain(store, &embedder, query, rowid, &plan, TOP_K)
                    })?;
                    assert_eq!(result, expected.explanation);
                    baseline.push(elapsed);
                }
                1 => {
                    let (elapsed, result) =
                        timed(|| without_mapping(store, &embedder, query, rowid, &plan))?;
                    assert_eq!(result, raw);
                    diagnostics.push(elapsed);
                }
                _ => {
                    let (elapsed, result) =
                        timed(|| explain_with_fts(store, &embedder, query, rowid, &plan, TOP_K))?;
                    assert_eq!(result, expected);
                    let details = result.lexical_details.matched.as_ref().unwrap();
                    max_error = max_error.max((details.bm25 - details.contribution_sum).abs());
                    complete.push(elapsed);
                }
            }
        }
        // Clone scoring facts outside the timer. The timer covers verification,
        // DOCUMENT tokenization and all original-text hit mapping, not scoring.
        let opened = reopen(path, strategy, original, reopened)?;
        let store = opened.as_ref().unwrap_or(original);
        let mut details = raw.1.clone();
        let started = Instant::now();
        store.map_fts_hits(&mut details, text);
        black_box(&details);
        mapping.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        assert_eq!(&details, matched);
    }
    assert_eq!(embedder.query_calls.get() - before_queries, repetitions * 3);
    let baseline = Distribution::new(baseline);
    let diagnostics = Distribution::new(diagnostics);
    let complete = Distribution::new(complete);
    Ok(Measurement {
        strategy: strategy.into(),
        corpus_rows: rows,
        storage: storage.into(),
        cache_condition: if reopened {
            "reopened_connection_first_query_os_cache_uncontrolled"
        } else {
            "warm_connection_and_os_cache"
        }
        .into(),
        query: query.into(),
        match_expression: expected.lexical_details.match_expression.unwrap(),
        target_rowid: rowid,
        target_bytes: text.len(),
        phrase_count: matched.phrases.len(),
        hit_count: hits.len(),
        in_candidate_pool: expected.lexical_details.in_candidate_pool,
        additional_diagnostics_median_us: diagnostics.median_us - baseline.median_us,
        additional_complete_median_us: complete.median_us - baseline.median_us,
        complete_minus_diagnostics_median_us: complete.median_us - diagnostics.median_us,
        baseline,
        diagnostics_without_mapping: diagnostics,
        complete,
        mapping_only: Distribution::new(mapping),
        max_bm25_error: max_error,
    })
}

fn matrix(output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        !black_box(cfg!(debug_assertions)),
        "Performance measurements require --release"
    );
    let workspace_scratch = output.parent().unwrap().join("scratch");
    std::fs::create_dir_all(&workspace_scratch)?;
    let directory = tempfile::tempdir_in(&workspace_scratch)?;
    let mut measurements = Vec::new();
    let sizes: Vec<u32> = std::env::var("LODE_FTS_PERFORMANCE_SIZES")
        .unwrap_or_else(|_| "1000,10000,100000".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    for strategy in ["unicode61", "trigram", "simple", "jieba"] {
        for &rows in &sizes {
            assert!(rows >= 2);
            for storage in ["memory", "disk"] {
                let path = if storage == "memory" {
                    Path::new(":memory:").to_owned()
                } else {
                    directory.path().join(format!("{strategy}-{rows}.db"))
                };
                let store = fixture::build(&path, strategy, rows)?;
                for rowid in [1, 2] {
                    for query in [
                        "common",
                        "needleabc",
                        "alpha beta common 知识 中国",
                        "alpha",
                    ] {
                        for reopened in [false, true] {
                            if reopened && storage == "memory" {
                                continue;
                            }
                            measurements.push(measure(
                                &store, &path, strategy, rows, storage, reopened, query, rowid,
                            )?);
                        }
                    }
                }
                eprintln!("measured {strategy}/{rows}/{storage}");
            }
        }
    }
    let compiler = Command::new("rustc").arg("--version").output()?;
    let report = serde_json::json!({
        "compiler": String::from_utf8(compiler.stdout)?.trim(),
        "sqlite": rusqlite::version(),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "release_build": !cfg!(debug_assertions),
        "top_k": TOP_K,
        "embedding": "fixed in-process FakeEmbedder, four-dimensional vectors, exactly one query embedding per complete pipeline",
        "sizes": sizes,
        "warm_samples": WARM_SAMPLES,
        "reopened_samples": REOPEN_SAMPLES,
        "public_cli_json_included": false,
        "auxiliary_private_json_encode_decode_included": true,
        "opening_schema_fixture_creation_and_result_checks_included": false,
        "mapping_clone_included": false,
        "true_disk_cold_cache_measured": false,
        "measurements": measurements,
    });
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

#[test]
#[ignore = "opt-in release performance measurement; no latency thresholds"]
fn production_performance() {
    let output = std::env::var_os("LODE_FTS_PERFORMANCE_REPORT")
        .expect("set LODE_FTS_PERFORMANCE_REPORT to an absolute report path");
    if std::env::var_os(CHILD_ENV).is_some() {
        matrix(Path::new(&output)).unwrap();
        return;
    }
    let cache = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
        .env(CHILD_ENV, "1")
        .env("LODE_CACHE_DIR", cache.path())
        .status()
        .unwrap();
    assert!(status.success(), "performance child failed: {status}");
}
