use crate::fts::MatchExpr;
use crate::fts::explanation::{FtsHitMapping, FtsMappingUnavailable};

#[test]
fn document_length_mismatch_withholds_ranges_without_changing_bm25() {
    let (_dir, store) = super::tests::fixture("unicode61");
    let expression = MatchExpr::Prebuilt("\"alpha\"".into());
    let mut details = store.explain_fts(&expression, 1).unwrap().matched.unwrap();
    let original_bm25 = details.bm25;
    store.map_fts_hits(&mut details, "wrong length");
    assert!(
        details
            .phrases
            .iter()
            .flat_map(|phrase| &phrase.hits)
            .all(|hit| matches!(
                hit.mapping,
                FtsHitMapping::Unavailable(FtsMappingUnavailable::TokenCountMismatch)
            ))
    );
    assert!((details.bm25 - original_bm25).abs() < f64::EPSILON);
}

use crate::embeddings::testing::{file_record, make_chunks};
use crate::fts::explanation::{FtsMatchDetails, FtsSourceSpan};
use crate::index::store::Store;

fn mapping_fixture(strategy: &str, text: &str) -> (tempfile::TempDir, Store) {
    let directory = tempfile::tempdir().unwrap();
    let mut store = Store::open(
        &directory.path().join("index.db"),
        "test-model",
        4,
        strategy,
    )
    .unwrap();
    let (chunks, vectors) = make_chunks(&[text], None);
    store
        .replace_file(
            &file_record("notes.txt", "blake3:mapping", 1),
            "text",
            &chunks,
            Some(&vectors),
        )
        .unwrap();
    (directory, store)
}

fn mapped(store: &Store, expression: &MatchExpr, text: &str) -> FtsMatchDetails {
    let mut details = store.explain_fts(expression, 1).unwrap().matched.unwrap();
    let original = details.clone();
    store.map_fts_hits(&mut details, text);
    let mut scores_only = details.clone();
    for hit in scores_only
        .phrases
        .iter_mut()
        .flat_map(|phrase| &mut phrase.hits)
    {
        hit.mapping = FtsHitMapping::default();
    }
    assert_eq!(
        scores_only, original,
        "mapping cannot change any scoring fact or document token"
    );
    for hit in details.phrases.iter().flat_map(|phrase| &phrase.hits) {
        let FtsHitMapping::Mapped(span) = &hit.mapping else {
            panic!(
                "mapping unavailable for valid DOCUMENT stream: {:?}",
                hit.mapping
            );
        };
        assert_eq!(
            text.get(span.start_byte..span.end_byte),
            Some(span.surface.as_str())
        );
        assert!(text.is_char_boundary(span.start_byte));
        assert!(text.is_char_boundary(span.end_byte));
    }
    details
}

fn surfaces(details: &FtsMatchDetails) -> Vec<&FtsSourceSpan> {
    details
        .phrases
        .iter()
        .flat_map(|phrase| &phrase.hits)
        .map(|hit| {
            let FtsHitMapping::Mapped(span) = &hit.mapping else {
                panic!("expected mapped");
            };
            span
        })
        .collect()
}

#[test]
fn unicode_combining_marks_normalization_repeats_and_prefixes_map_to_original_text() {
    let text = "🙂 Café, CAFÉ\na\u{301}b 中文";
    let (_dir, store) = mapping_fixture("unicode61", text);
    let details = mapped(
        &store,
        &MatchExpr::Prebuilt("\"cafe\" OR \"ab\" OR \"中文\"".into()),
        text,
    );
    assert_eq!(
        surfaces(&details)
            .iter()
            .map(|span| span.surface.as_str())
            .collect::<Vec<_>>(),
        ["Café", "CAFÉ", "a\u{301}b", "中文"]
    );
    let phrase = mapped(&store, &MatchExpr::Prebuilt("\"cafe ab\"".into()), text);
    assert_eq!(surfaces(&phrase)[0].surface, "CAFÉ\na\u{301}b");
    assert_eq!(phrase.phrases[0].query_tokens, ["cafe", "ab"]);
    let prefix = mapped(&store, &MatchExpr::Prebuilt("\"caf\"*".into()), text);
    assert_eq!(
        surfaces(&prefix)
            .iter()
            .map(|span| span.surface.as_str())
            .collect::<Vec<_>>(),
        ["Café", "CAFÉ"]
    );
    assert!(
        prefix.phrases[0]
            .hits
            .iter()
            .all(|hit| hit.document_tokens == vec![None])
    );
}

#[test]
fn overlapping_trigram_and_multi_token_phrases_cover_complete_original_slices() {
    let text = "Ab中文🙂 Ab中文🙂 知识知识知识";
    let (_dir, store) = mapping_fixture("trigram", text);
    let repeated = mapped(
        &store,
        &MatchExpr::Prebuilt("\"Ab中\" OR \"b中文\"".into()),
        text,
    );
    let spans = surfaces(&repeated);
    assert_eq!(spans.len(), 4);
    assert!(spans[0].end_byte > spans[2].start_byte);
    let phrase = mapped(&store, &MatchExpr::Prebuilt("\"Ab中文\"".into()), text);
    assert_eq!(phrase.phrases[0].query_tokens.len(), 2);
    assert_eq!(
        surfaces(&phrase)
            .iter()
            .map(|span| span.surface.as_str())
            .collect::<Vec<_>>(),
        ["Ab中文", "Ab中文"]
    );
    let chinese = mapped(&store, &MatchExpr::Prebuilt("\"知识知识\"".into()), text);
    assert_eq!(surfaces(&chinese).len(), 2);
    assert!(
        surfaces(&chinese)
            .iter()
            .all(|span| span.surface == "知识知识")
    );
}

#[test]
fn native_pinyin_polyphonic_and_jieba_phrases_share_document_ranges() {
    crate::run_with_isolated_jieba_cache(
        "index::store::fts::mapping_tests::native_pinyin_polyphonic_and_jieba_phrases_share_document_ranges",
        || {
            let text = "重庆 重复重复 中国";
            for strategy in ["simple", "jieba"] {
                let (_dir, store) = mapping_fixture(strategy, text);
                let variants = mapped(
                    &store,
                    &MatchExpr::Prebuilt("\"重\" OR \"zhong\" OR \"chong\"".into()),
                    text,
                );
                assert!(
                    surfaces(&variants)
                        .iter()
                        .all(|span| matches!(span.surface.as_str(), "重" | "中"))
                );
                let starts: Vec<_> = surfaces(&variants)
                    .iter()
                    .map(|span| span.start_byte)
                    .collect();
                for start in [0, 7, 13] {
                    assert_eq!(starts.iter().filter(|value| **value == start).count(), 3);
                }
                let phrase = mapped(&store, &MatchExpr::Prebuilt("\"chong qing\"".into()), text);
                assert_eq!(surfaces(&phrase)[0].surface, "重庆");
                let actual = mapped(
                    &store,
                    &crate::fts::match_query("中国", strategy).unwrap(),
                    text,
                );
                assert!(
                    surfaces(&actual)
                        .iter()
                        .all(|span| matches!(span.surface.as_str(), "中" | "国" | "中国"))
                );
                let complete = mapped(&store, &MatchExpr::Prebuilt("\"中国\"".into()), text);
                assert_eq!(surfaces(&complete)[0].surface, "中国");
            }
        },
    );
}

#[test]
fn unknown_tokenizer_parameters_and_native_nul_input_withhold_ranges() {
    let (_dir, store) = mapping_fixture("unicode61", "alpha Café");
    store.conn.execute_batch("DROP TABLE chunks_fts;").unwrap();
    store
        .conn
        .execute_batch(&crate::index::schema::fts_table_sql(
            "unicode61 remove_diacritics 0",
        ))
        .unwrap();
    store
        .conn
        .execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')", [])
        .unwrap();
    let mut details = store
        .explain_fts(&MatchExpr::Prebuilt("\"alpha\"".into()), 1)
        .unwrap()
        .matched
        .unwrap();
    let original_bm25 = details.bm25;
    store.map_fts_hits(&mut details, "alpha Café");
    assert!(
        details
            .phrases
            .iter()
            .flat_map(|phrase| &phrase.hits)
            .all(|hit| matches!(
                hit.mapping,
                FtsHitMapping::Unavailable(FtsMappingUnavailable::TokenizerConfiguration)
            ))
    );
    assert!((original_bm25 - details.bm25).abs() < f64::EPSILON);
    let (_dir, store) = mapping_fixture("simple", "alpha\0beta");
    let mut details = store
        .explain_fts(&MatchExpr::Prebuilt("\"alpha\"".into()), 1)
        .unwrap()
        .matched
        .unwrap();
    store.map_fts_hits(&mut details, "alpha\0beta");
    assert!(matches!(
        details.phrases[0].hits[0].mapping,
        FtsHitMapping::Unavailable(FtsMappingUnavailable::TokenizationFailed)
    ));
}
