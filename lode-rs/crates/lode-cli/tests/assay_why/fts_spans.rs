#![warn(clippy::pedantic)]

use super::fts_details::lexical_config;
use super::*;

fn details(fixture: &Fixture, query: &str) -> Value {
    let output = run_lode(
        fixture,
        "json",
        &["assay", "why", &fixture.digest, query, "--fts-details"],
    );
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    json(&output)
}

fn hits(payload: &Value) -> Vec<&Value> {
    payload["lexical_details"]["phrases"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|phrase| phrase["hits"].as_array().unwrap())
        .collect()
}

fn assert_mapped(payload: &Value, text: &str) {
    assert_eq!(payload["lexical_details"]["status"], "matched");
    assert!(!hits(payload).is_empty());
    for hit in hits(payload) {
        assert_eq!(hit["mapping_status"], "mapped");
        assert!(hit["mapping_reason"].is_null());
        let start = usize::try_from(hit["start_byte"].as_u64().unwrap()).unwrap();
        let end = usize::try_from(hit["end_byte"].as_u64().unwrap()).unwrap();
        assert!(start < end);
        assert!(text.is_char_boundary(start));
        assert!(text.is_char_boundary(end));
        assert_eq!(text.get(start..end), hit["surface"].as_str());
        assert_eq!(hit.as_object().unwrap().len(), 8);
    }
    let details = &payload["lexical_details"];
    let contribution_sum: f64 = details["phrases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|phrase| phrase["bm25_contribution"].as_f64().unwrap())
        .sum();
    assert!((contribution_sum - details["bm25"].as_f64().unwrap()).abs() < 1e-12);
    assert!(payload.get("text").is_none());
}

#[test]
fn unicode_normalization_and_combining_characters_keep_original_spelling_and_byte_bounds() {
    let text = "🙂 Café CAFÉ\na\u{301}b 中文";
    let fixture = fixture(&lexical_config("unicode61"), text);
    let payload = details(&fixture, "cafe 中文");
    assert_mapped(&payload, text);
    let surfaces: Vec<_> = hits(&payload)
        .iter()
        .map(|hit| hit["surface"].as_str().unwrap())
        .collect();
    assert_eq!(surfaces, ["Café", "CAFÉ", "中文"]);
    assert_eq!(hits(&payload)[0]["document_tokens"][0], "cafe");
    let combining = details(&fixture, "ab");
    assert_mapped(&combining, text);
    assert_eq!(hits(&combining)[0]["surface"], "a\u{301}b");
    let output = run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "cafe", "--fts-details"],
    );
    assert!(output.status.success());
    let output = String::from_utf8_lossy(&output.stdout);
    assert!(output.contains("source mapping: mapped"));
    assert!(output.contains("surface=\"Café\""));
}

#[test]
fn repeated_trigram_hits_keep_overlapping_original_ranges() {
    let text = "🙂 Ab中文🙂 Ab中文🙂";
    let fixture = fixture_with_physical_tokenizer(&lexical_config("trigram"), text, "trigram");
    let payload = details(&fixture, "Ab中文");
    assert_mapped(&payload, text);
    let all = hits(&payload);
    assert_eq!(all.len(), 4);
    assert!(all[0]["end_byte"].as_u64().unwrap() > all[2]["start_byte"].as_u64().unwrap());
    for hit in all {
        assert_eq!(hit["surface"].as_str().unwrap().chars().count(), 3);
    }
}

#[test]
fn native_pinyin_and_latin_prefixes_keep_unknown_document_tokens_and_real_surfaces() {
    let text = "重庆 重复重复 中国 ZHONGWEN";
    for strategy in ["simple", "jieba"] {
        let fixture = fixture_with_physical_tokenizer(&lexical_config(strategy), text, "simple");
        let payload = details(&fixture, "zhong");
        assert_mapped(&payload, text);
        let all = hits(&payload);
        assert!(all.iter().any(|hit| hit["surface"] == "重"));
        assert!(all.iter().any(|hit| hit["surface"] == "中"));
        assert!(all.iter().any(|hit| hit["surface"] == "ZHONGWEN"));
        assert!(all.iter().any(|hit| {
            hit["document_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .any(Value::is_null)
        }));
        let output = run_lode(
            &fixture,
            "extended",
            &["analyze", "why", &fixture.digest, "zhong", "--fts-details"],
        );
        assert!(output.status.success());
        let output = String::from_utf8_lossy(&output.stdout);
        assert!(output.contains("unknown document token"));
        assert!(output.contains("surface=\"ZHONGWEN\""));
    }
}

#[test]
fn native_helpers_preserve_actual_phrase_ranges_and_colocated_surfaces() {
    let text = "中国 中国 知识";
    for strategy in ["simple", "jieba"] {
        let fixture = fixture_with_physical_tokenizer(&lexical_config(strategy), text, "simple");
        let payload = details(&fixture, "中国");
        assert_mapped(&payload, text);
        let all = hits(&payload);
        let ranges: Vec<_> = all
            .iter()
            .map(|hit| {
                (
                    hit["start_byte"].as_u64().unwrap(),
                    hit["end_byte"].as_u64().unwrap(),
                )
            })
            .collect();
        if strategy == "jieba" {
            assert!(
                payload["lexical_details"]["phrases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|phrase| {
                        phrase["query_tokens"].as_array().unwrap().len() > 1
                            && !phrase["hits"].as_array().unwrap().is_empty()
                    }),
                "{strategy}: {payload}"
            );
            assert!(all.iter().all(|hit| hit["surface"] == "中国"));
            assert!(ranges.contains(&(0, 6)));
            assert!(ranges.contains(&(7, 13)));
        } else {
            assert!(
                all.iter()
                    .all(|hit| matches!(hit["surface"].as_str(), Some("中" | "国")))
            );
            for range in [(0, 3), (3, 6), (7, 10), (10, 13)] {
                assert!(ranges.contains(&range), "{strategy}: {payload}");
            }
        }
    }
}

#[test]
fn unknown_configuration_retains_scores_and_reports_unavailable_ranges() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha Café");
    let before = details(&fixture, "alpha");
    let connection = rusqlite::Connection::open(fixture.workspace.join(".lode/index.db")).unwrap();
    connection.execute_batch(
        "DROP TABLE chunks_fts;
         CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61 remove_diacritics 0', content='chunks', content_rowid='id');
         INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');"
    ).unwrap();
    drop(connection);
    let payload = details(&fixture, "alpha");
    assert_eq!(payload["sources"], before["sources"]);
    assert_eq!(payload["combined"], before["combined"]);
    assert_eq!(
        payload["lexical_details"]["bm25"],
        before["lexical_details"]["bm25"]
    );
    for hit in hits(&payload) {
        assert_eq!(hit["mapping_status"], "unavailable");
        assert_eq!(hit["mapping_reason"], "tokenizer_configuration");
        for field in ["start_byte", "end_byte", "surface"] {
            assert!(hit[field].is_null());
        }
    }
    let output = run_lode(
        &fixture,
        "extended",
        &["assay", "why", &fixture.digest, "alpha", "--fts-details"],
    );
    assert!(output.status.success());
    let output = String::from_utf8_lossy(&output.stdout);
    assert!(output.contains("source mapping: unavailable"));
    assert!(output.contains("tokenizer configuration could not be verified"));
    assert!(output.contains("raw BM25 (display rounded):"));
}

#[test]
fn invalid_indexed_document_size_never_produces_a_guessed_range() {
    let fixture = fixture(&lexical_config("unicode61"), "alpha beta");
    let connection = rusqlite::Connection::open(fixture.workspace.join(".lode/index.db")).unwrap();
    connection
        .execute("UPDATE chunks_fts_docsize SET sz=x'03' WHERE id=1", [])
        .unwrap();
    drop(connection);
    let payload = details(&fixture, "alpha");
    assert_eq!(payload["lexical_details"]["status"], "matched");
    assert!(payload["lexical_details"]["bm25"].is_number());
    assert_eq!(payload["lexical_details"]["document_length"], 3);
    for hit in hits(&payload) {
        assert_eq!(hit["mapping_status"], "unavailable");
        assert_eq!(hit["mapping_reason"], "token_count_mismatch");
        assert!(hit["start_byte"].is_null());
        assert!(hit["end_byte"].is_null());
        assert!(hit["surface"].is_null());
    }
}

#[test]
fn stale_prefix_ranges_refer_to_indexed_chunk_rather_than_changed_file() {
    let fixture =
        fixture_with_physical_tokenizer(&lexical_config("simple"), "alphabet 中文", "simple");
    fs::write(fixture.workspace.join("notes.txt"), "changed quartz source").unwrap();
    let payload = details(&fixture, "alpha");
    assert_mapped(&payload, "alphabet 中文");
    assert_eq!(payload["has_stale"], true);
    assert!(
        hits(&payload)
            .iter()
            .all(|hit| hit["surface"] == "alphabet")
    );
}
