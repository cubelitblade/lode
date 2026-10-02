use super::*;

#[test]
fn schema_identity_keeps_tokenizer_arguments_and_quoted_whitespace() {
    let expected =
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61', content='chunks');";
    assert!(same_definition(
        "create virtual table chunks_fts using fts5 ( text , tokenize = 'unicode61', content = 'chunks' )",
        expected
    ));
    for altered in [
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61 remove_diacritics 0', content='chunks');",
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61', content='chu nks');",
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61', content='chunks', detail=none);",
    ] {
        assert!(!same_definition(altered, expected));
    }
}

#[test]
fn colocated_tokens_share_ranges_and_invalid_streams_are_rejected() {
    let token = |flags, start, end| Token {
        flags,
        text: "中".into(),
        start,
        end,
    };
    let grouped = positions(
        "中国",
        vec![token(0, 0, 3), token(TOKEN_COLOCATED, 0, 3), token(0, 3, 6)],
    )
    .unwrap();
    assert_eq!(grouped.len(), 2);
    assert_eq!(grouped[0].variants.len(), 2);
    for tokens in [
        vec![token(TOKEN_COLOCATED, 0, 3)],
        vec![token(2, 0, 3)],
        vec![token(0, 0, 1)],
        vec![token(0, 0, 99)],
        vec![token(0, 3, 6), token(0, 0, 3)],
        vec![token(0, 0, 3), token(TOKEN_COLOCATED, 3, 6)],
    ] {
        assert!(matches!(
            positions("中国", tokens),
            Err(FtsMappingUnavailable::InvalidTokenStream)
        ));
    }
}

#[test]
fn per_hit_failures_withhold_ranges_for_bad_offsets_columns_or_index_tokens() {
    let positions = positions(
        "alpha beta",
        vec![
            Token {
                flags: 0,
                text: "alpha".into(),
                start: 0,
                end: 5,
            },
            Token {
                flags: 0,
                text: "beta".into(),
                start: 6,
                end: 10,
            },
        ],
    )
    .unwrap();
    let mut hit = FtsHit {
        column: 0,
        token_offset: 0,
        document_tokens: vec![Some("alpha".into())],
        mapping: FtsHitMapping::default(),
    };
    assert_eq!(
        map_hit("alpha beta", &positions, &hit, 1).unwrap().surface,
        "alpha"
    );
    hit.column = 1;
    assert_eq!(
        map_hit("alpha beta", &positions, &hit, 1),
        Err(FtsMappingUnavailable::UnsupportedColumn)
    );
    hit.column = 0;
    hit.token_offset = i32::MAX;
    assert_eq!(
        map_hit("alpha beta", &positions, &hit, 1),
        Err(FtsMappingUnavailable::HitOutsideDocument)
    );
    hit.token_offset = 0;
    hit.document_tokens = vec![Some("wrong".into())];
    assert_eq!(
        map_hit("alpha beta", &positions, &hit, 1),
        Err(FtsMappingUnavailable::DocumentTokenMismatch)
    );
    hit.document_tokens = vec![None];
    assert_eq!(
        map_hit("alpha beta", &positions, &hit, 1).unwrap().surface,
        "alpha"
    );
}
