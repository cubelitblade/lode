#![warn(clippy::pedantic)]

//! Run with `cargo run --release -p lode-core --example fts_bm25_probe`.
//! Synthetic indexes only; no workspace index or embedding requests.

mod auxiliary;

use std::error::Error;
use std::fmt::Write as _;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use auxiliary::Explanation;
use lode_core::fts::{MatchExpr, explain_query, match_query};
use rusqlite::{Connection, params};
use serde::Serialize;

type ProbeResult<T> = Result<T, Box<dyn Error>>;
const CHILD_ENV: &str = "LODE_FTS_PROBE_CHILD";
const ABSOLUTE_TOLERANCE: f64 = 1e-12;
const RELATIVE_TOLERANCE: f64 = 1e-12;
const TIMING_REPETITIONS: usize = 21;
const DOCUMENTS: &[&str] = &[
    "alpha beta alpha alphabet 知识 中国 common",
    "alpha gamma 知识库 知识 common",
    "beta gamma 中国人 common",
    "alpha beta gamma common",
    "knowledge mining document retrieval common",
    "中华人民共和国 知识 common",
    "重庆 重复重复 知识 common",
    "alphabet alphanumeric beta common",
    "café résumé common",
    "alpha alpha alpha common",
    "alpha far far far far beta common",
];

#[derive(Serialize)]
struct RowEvidence {
    bm25: f64,
    contribution_sum: f64,
    absolute_error: f64,
    explanation: Explanation,
}

#[derive(Serialize)]
struct Case {
    strategy: String,
    name: String,
    user_query: Option<String>,
    match_expression: String,
    column_weights: Vec<f64>,
    rejected_match: Option<String>,
    target_filter_verified: bool,
    rows: Vec<RowEvidence>,
}

#[derive(Serialize)]
struct Timing {
    strategy: String,
    corpus_rows: usize,
    query: String,
    match_expression: String,
    repetitions: usize,
    target_rowid: i64,
    baseline_median_us: f64,
    with_explanation_median_us: f64,
    additional_median_us: f64,
    ratio: f64,
}

fn connection(strategy: &str, rows: usize, multicolumn: bool) -> ProbeResult<Connection> {
    let connection = Connection::open_in_memory()?;
    if matches!(strategy, "simple" | "jieba") {
        lode_simple_native::register_connection(&connection, strategy == "jieba")?;
    }
    auxiliary::register(&connection)?;
    let physical = if strategy == "jieba" {
        "simple"
    } else {
        strategy
    };
    let columns = if multicolumn { "text, title" } else { "text" };
    // Lode's external-content layout; extra columns exercise API weights
    // separately from the current single-column production schema.
    connection.execute_batch(&format!(
        "CREATE TABLE chunks(id INTEGER PRIMARY KEY, text TEXT NOT NULL, title TEXT NOT NULL);
         CREATE VIRTUAL TABLE chunks_fts USING fts5({columns}, tokenize='{physical}', content='chunks', content_rowid='id');
         BEGIN;"
    ))?;
    for index in 0..rows {
        let text = if rows <= 100 {
            DOCUMENTS.get(index).map_or_else(
                || format!("filler{index} unrelated common"),
                |text| (*text).to_string(),
            )
        } else {
            // Frequent phrases force a corpus-wide document-frequency pass.
            format!("alpha beta alpha alphabet 知识 中国 common filler{index}")
        };
        let rowid = i64::try_from(index + 1)?;
        let title = if index % 2 == 0 {
            "alpha 知识"
        } else {
            "beta 中国"
        };
        connection.execute(
            "INSERT INTO chunks VALUES (?1, ?2, ?3)",
            params![rowid, text, title],
        )?;
        if multicolumn {
            connection.execute(
                "INSERT INTO chunks_fts(rowid, text, title) VALUES (?1, ?2, ?3)",
                params![rowid, text, title],
            )?;
        } else {
            connection.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?1, ?2)",
                params![rowid, text],
            )?;
        }
    }
    connection.execute_batch("COMMIT;")?;
    Ok(connection)
}

fn construct(connection: &Connection, strategy: &str, query: &str) -> ProbeResult<String> {
    let expression = match match_query(query, strategy)? {
        MatchExpr::Prebuilt(expression) => expression,
        MatchExpr::Helper { function, query } => connection
            .query_row(&format!("SELECT {function}(?1)"), [query], |row| {
                row.get::<_, Option<String>>(0)
            })?
            .unwrap_or_default(),
    };
    assert_eq!(expression, explain_query(query, strategy)?.match_expression);
    Ok(expression)
}

fn run_case(
    connection: &Connection,
    strategy: &str,
    name: &str,
    query: Option<&str>,
    expression: &str,
    weights: &[f64],
) -> ProbeResult<Case> {
    let changes = connection.total_changes();
    let mut evidence: Vec<RowEvidence> = Vec::new();
    let mut target_filter_verified = false;
    if !expression.is_empty() {
        let mut arguments = String::new();
        for weight in weights {
            write!(&mut arguments, ", {weight}")?;
        }
        let mut statement = connection.prepare(&format!(
            "SELECT rowid, bm25(chunks_fts{arguments}), phrase_probe(chunks_fts{arguments})
             FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY rowid"
        ))?;
        let mut rows = statement.query([expression])?;
        while let Some(row) = rows.next()? {
            let bm25: f64 = row.get(1)?;
            let explanation: Explanation = serde_json::from_str(&row.get::<_, String>(2)?)?;
            assert_eq!(explanation.rowid, row.get::<_, i64>(0)?);
            let contribution_sum: f64 = explanation
                .phrases
                .iter()
                .map(|phrase| phrase.bm25_contribution)
                .sum();
            let absolute_error = (bm25 - contribution_sum).abs();
            assert!(bm25.is_finite() && contribution_sum.is_finite());
            assert!(
                absolute_error <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * bm25.abs(),
                "{strategy}/{name}/{}: {bm25} != {contribution_sum}",
                explanation.rowid
            );
            for phrase in &explanation.phrases {
                assert!(
                    phrase.document_frequency >= 0
                        && phrase.document_frequency <= explanation.corpus_rows
                );
                for hit in &phrase.hits {
                    assert!(hit.token_offset >= 0);
                    assert_eq!(hit.document_tokens.len(), phrase.query_tokens.len());
                }
            }
            evidence.push(RowEvidence {
                bm25,
                contribution_sum,
                absolute_error,
                explanation,
            });
        }
        if let Some(target) = evidence.first() {
            let (bm25, json): (f64, String) = connection.query_row(
                &format!(
                    "SELECT bm25(chunks_fts{arguments}), phrase_probe(chunks_fts{arguments})
                          FROM chunks_fts WHERE chunks_fts MATCH ?1 AND rowid=?2"
                ),
                params![expression, target.explanation.rowid],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            let filtered: Explanation = serde_json::from_str(&json)?;
            assert!((bm25 - target.bm25).abs() <= ABSOLUTE_TOLERANCE);
            assert_eq!(filtered.corpus_rows, target.explanation.corpus_rows);
            let sum: f64 = filtered
                .phrases
                .iter()
                .map(|phrase| phrase.bm25_contribution)
                .sum();
            assert!((sum - bm25).abs() <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * bm25.abs());
            assert_eq!(
                filtered
                    .phrases
                    .iter()
                    .map(|phrase| phrase.document_frequency)
                    .collect::<Vec<_>>(),
                target
                    .explanation
                    .phrases
                    .iter()
                    .map(|phrase| phrase.document_frequency)
                    .collect::<Vec<_>>()
            );
            target_filter_verified = true;
        }
    }
    assert_eq!(
        connection.total_changes(),
        changes,
        "diagnostics changed the fixture"
    );
    Ok(Case {
        strategy: strategy.to_string(),
        name: name.to_string(),
        user_query: query.map(str::to_string),
        match_expression: expression.to_string(),
        column_weights: weights.to_vec(),
        rejected_match: None,
        target_filter_verified,
        rows: evidence,
    })
}

fn column_cases(strategy: &str) -> ProbeResult<Vec<Case>> {
    let mut cases = Vec::new();
    let multi = connection(strategy, 31, true)?;
    for (name, expression, weights) in [
        ("column_weights", "\"alpha\" OR \"beta\"", &[3.0, 0.25][..]),
        ("column_filter", "title : \"alpha\"", &[3.0, 0.25][..]),
        ("zero_weights", "\"alpha\"", &[0.0, 0.0][..]),
    ] {
        let case = run_case(&multi, strategy, name, None, expression, weights)?;
        assert!(!case.rows.is_empty());
        if name == "column_filter" {
            assert!(
                case.rows
                    .iter()
                    .flat_map(|row| &row.explanation.phrases)
                    .flat_map(|phrase| &phrase.hits)
                    .all(|hit| hit.column == 1)
            );
            assert_eq!(case.rows[0].explanation.phrases[0].document_frequency, 16);
        }
        cases.push(case);
    }
    Ok(cases)
}

fn constructed_case(
    conn: &Connection,
    strategy: &str,
    query: &str,
    expression: &str,
) -> ProbeResult<Case> {
    if expression.is_empty() {
        return run_case(conn, strategy, "constructed", Some(query), expression, &[]);
    }
    let baseline = conn
        .prepare("SELECT bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?1")?
        .query_map([&expression], |row| row.get::<_, f64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>();
    if let Err(error) = baseline {
        assert!(
            strategy == "trigram" && query == "alpha \" beta",
            "unexpected baseline error: {strategy}/{query}: {error}"
        );
        let diagnostic_error =
            run_case(conn, strategy, "constructed", Some(query), expression, &[])
                .err()
                .ok_or("diagnostic unexpectedly accepted invalid MATCH")?;
        assert_eq!(error.to_string(), diagnostic_error.to_string());
        Ok(Case {
            strategy: strategy.to_string(),
            name: "constructed_rejected".to_string(),
            user_query: Some(query.to_string()),
            match_expression: expression.to_string(),
            column_weights: Vec::new(),
            rejected_match: Some(error.to_string()),
            target_filter_verified: false,
            rows: Vec::new(),
        })
    } else {
        run_case(conn, strategy, "constructed", Some(query), expression, &[])
    }
}

fn raw_cases(conn: &Connection, strategy: &str) -> ProbeResult<Vec<Case>> {
    let mut cases = Vec::new();
    let (first, second, missing) = if strategy == "trigram" {
        ("alp", "bet", "zzz")
    } else {
        ("alpha", "beta", "absentword")
    };
    for (name, expression) in [
        ("or", format!("\"{first}\" OR \"{second}\"")),
        ("and", format!("\"{first}\" AND \"{second}\"")),
        ("phrase", "\"alpha beta\"".to_string()),
        ("prefix", "alph*".to_string()),
        ("repeated_phrase", format!("\"{first}\" OR \"{first}\"")),
        ("not", format!("\"{first}\" NOT \"gamma\"")),
        ("near", "NEAR(alpha beta, 10)".to_string()),
        ("unmatched_branch", format!("\"{first}\" OR \"{missing}\"")),
        ("idf_floor", "\"common\"".to_string()),
        ("no_match", format!("\"{missing}\"")),
    ] {
        let case = run_case(conn, strategy, name, None, &expression, &[])?;
        assert_eq!(
            case.rows.is_empty(),
            name == "no_match",
            "{strategy}/{name}"
        );
        if name == "phrase" {
            assert!(
                case.rows
                    .iter()
                    .all(|row| row.explanation.phrases.len() == 1
                        && row.explanation.phrases[0].query_tokens.len() > 1)
            );
        }
        if name == "repeated_phrase" {
            let phrases = &case.rows[0].explanation.phrases;
            assert_eq!(phrases.len(), 2);
            assert_eq!(phrases[0].query_tokens, phrases[1].query_tokens);
            let expected_frequency = if strategy == "trigram" { 3.0 } else { 2.0 };
            assert!((phrases[0].weighted_frequency - expected_frequency).abs() < f64::EPSILON);
            assert!(
                (phrases[0].bm25_contribution - phrases[1].bm25_contribution).abs()
                    <= ABSOLUTE_TOLERANCE
            );
        }
        if name == "unmatched_branch" {
            let phrase = &case.rows[0].explanation.phrases[1];
            assert_eq!(phrase.document_frequency, 0);
            assert!(phrase.hits.is_empty());
            assert!(phrase.bm25_contribution.abs() < f64::EPSILON);
        }
        if name == "prefix" {
            assert!(
                case.rows[0].explanation.phrases[0]
                    .hits
                    .iter()
                    .all(|hit| hit.document_tokens.last() == Some(&None))
            );
        }
        if name == "idf_floor" {
            assert!(
                case.rows
                    .iter()
                    .flat_map(|row| &row.explanation.phrases)
                    .all(|phrase| (phrase.idf - 1e-6).abs() < f64::EPSILON)
            );
        }
        cases.push(case);
    }
    Ok(cases)
}

fn correctness() -> ProbeResult<Vec<Case>> {
    let mut cases = Vec::new();
    for strategy in ["unicode61", "trigram", "simple", "jieba"] {
        let conn = connection(strategy, 31, false)?;
        for query in [
            "alpha beta",
            "知识",
            "zhishi",
            "中华人民共和国",
            "alpha alpha",
            "ALPHA",
            "café",
            "alpha \" beta",
            "absentword",
            "a",
        ] {
            let expression = construct(&conn, strategy, query)?;
            cases.push(constructed_case(&conn, strategy, query, &expression)?);
        }
        cases.extend(raw_cases(&conn, strategy)?);
        if matches!(strategy, "simple" | "jieba") {
            for (name, expression) in [
                ("pinyin", "\"zhi\" AND \"shi\""),
                ("colocated_or", "\"知\" OR \"zhi\""),
                ("chinese_phrase", "\"知识\""),
            ] {
                let case = run_case(&conn, strategy, name, None, expression, &[])?;
                assert!(!case.rows.is_empty());
                if name == "colocated_or" {
                    let phrases = &case.rows[0].explanation.phrases;
                    assert_eq!(
                        phrases[0].hits[0].token_offset,
                        phrases[1].hits[0].token_offset
                    );
                }
                cases.push(case);
            }
        }
        cases.extend(column_cases(strategy)?);
    }
    Ok(cases)
}

fn measure(
    statement: &mut rusqlite::Statement<'_>,
    expression: &str,
    explain: bool,
) -> ProbeResult<f64> {
    let started = Instant::now();
    statement.query_row([expression], |row| {
        black_box(row.get::<_, f64>(0)?);
        if explain {
            black_box(row.get::<_, String>(1)?);
        }
        Ok(())
    })?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn performance() -> ProbeResult<Vec<Timing>> {
    let mut timings = Vec::new();
    for strategy in ["unicode61", "trigram", "simple", "jieba"] {
        for corpus_rows in [1_000, 10_000] {
            let conn = connection(strategy, corpus_rows, false)?;
            let query = if strategy == "trigram" {
                "alpha beta"
            } else {
                "alpha 知识"
            };
            let expression = construct(&conn, strategy, query)?;
            let mut baseline = conn.prepare(
                "SELECT bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?1 AND rowid=1",
            )?;
            let mut diagnostic = conn.prepare("SELECT bm25(chunks_fts), phrase_probe(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?1 AND rowid=1")?;
            measure(&mut baseline, &expression, false)?;
            measure(&mut diagnostic, &expression, true)?;
            let mut baseline_times = Vec::new();
            let mut diagnostic_times = Vec::new();
            // Alternate order to reduce drift. Each execution resets FTS5's
            // statistics; pages and the native dictionary remain warm.
            for repetition in 0..TIMING_REPETITIONS {
                if repetition % 2 == 0 {
                    baseline_times.push(measure(&mut baseline, &expression, false)?);
                    diagnostic_times.push(measure(&mut diagnostic, &expression, true)?);
                } else {
                    diagnostic_times.push(measure(&mut diagnostic, &expression, true)?);
                    baseline_times.push(measure(&mut baseline, &expression, false)?);
                }
            }
            let baseline_median_us = median(&mut baseline_times);
            let with_explanation_median_us = median(&mut diagnostic_times);
            timings.push(Timing {
                strategy: strategy.to_string(),
                corpus_rows,
                query: query.to_string(),
                match_expression: expression,
                repetitions: TIMING_REPETITIONS,
                target_rowid: 1,
                baseline_median_us,
                with_explanation_median_us,
                additional_median_us: with_explanation_median_us - baseline_median_us,
                ratio: with_explanation_median_us / baseline_median_us,
            });
        }
    }
    Ok(timings)
}

fn main() -> ProbeResult<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        // The parent owns cleanup and overrides even an invalid user cache.
        let cache = tempfile::tempdir()?;
        let status = Command::new(std::env::current_exe()?)
            .env(CHILD_ENV, "1")
            .env("LODE_CACHE_DIR", cache.path())
            .status()?;
        if !status.success() {
            return Err(format!("probe child failed: {status}").into());
        }
        return Ok(());
    }
    let cases = correctness()?;
    let matched_rows: usize = cases.iter().map(|case| case.rows.len()).sum();
    let max_absolute_error = cases
        .iter()
        .flat_map(|case| &case.rows)
        .map(|row| row.absolute_error)
        .fold(0.0_f64, f64::max);
    let compiler = Command::new("rustc").arg("--version").output()?;
    let report = serde_json::json!({
        "compiler": String::from_utf8(compiler.stdout)?.trim(), "sqlite": rusqlite::version(),
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH, "release_build": !cfg!(debug_assertions),
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "case_count": cases.len(), "matched_rows": matched_rows, "max_absolute_error": max_absolute_error,
        "cases": cases, "timings": performance()?,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
