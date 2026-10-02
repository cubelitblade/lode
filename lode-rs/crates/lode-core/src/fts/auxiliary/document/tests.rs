use super::*;

#[test]
fn builtin_document_callbacks_preserve_unicode_and_overlapping_ranges() {
    let connection = Connection::open_in_memory().unwrap();
    let text = "🙂 CAFÉ a\u{301}b 中文";
    let tokens = builtin_document_tokens(&connection, c"unicode61", text).unwrap();
    assert_eq!(
        tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>(),
        ["🙂", "cafe", "ab", "中文"]
    );
    assert_eq!(&text[tokens[1].start..tokens[1].end], "CAFÉ");
    assert_eq!(&text[tokens[2].start..tokens[2].end], "a\u{301}b");
    let tokens = builtin_document_tokens(&connection, c"trigram", "A中文🙂Z").unwrap();
    assert_eq!(tokens.len(), 3);
    assert!(tokens[0].end > tokens[1].start);
    for token in tokens {
        assert_eq!("A中文🙂Z"[token.start..token.end].chars().count(), 3);
    }
}

#[test]
fn callback_failures_never_claim_invalid_utf8_or_out_of_bounds_ranges() {
    for (flags, bytes, start, end) in [
        (-1, b"x".as_slice(), 0, 1),
        (0, b"x".as_slice(), -1, 1),
        (0, b"x".as_slice(), 1, 2),
        (0, b"x".as_slice(), 0, 99),
        (0, b"\xff".as_slice(), 0, 3),
        (0, b"".as_slice(), 0, 3),
    ] {
        let mut collector = Collector {
            input: "中x",
            tokens: Vec::new(),
            error: None,
        };
        let code = unsafe {
            collect_token(
                (&raw mut collector).cast(),
                flags,
                bytes.as_ptr().cast(),
                c_int::try_from(bytes.len()).unwrap(),
                start,
                end,
            )
        };
        assert_ne!(code, ffi::SQLITE_OK);
        assert!(collector.error.is_some());
        assert!(collector.tokens.is_empty());
    }
    assert_eq!(
        unsafe { collect_token(ptr::null_mut(), 0, c"x".as_ptr(), 1, 0, 1) },
        ffi::SQLITE_MISUSE
    );
}

#[test]
fn tokenizer_lookup_failure_and_repeated_queries_release_resources() {
    let connection = Connection::open_in_memory().unwrap();
    assert!(builtin_document_tokens(&connection, c"missing-tokenizer", "alpha").is_err());
    for _ in 0..100 {
        assert_eq!(
            builtin_document_tokens(&connection, c"unicode61", "alpha beta")
                .unwrap()
                .len(),
            2
        );
    }
}

thread_local! {
    static INSTANCES: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

unsafe extern "C" fn create_probe(
    _: *mut c_void,
    _: *mut *const c_char,
    _: c_int,
    output: *mut *mut ffi::Fts5Tokenizer,
) -> c_int {
    unsafe {
        *output = Box::into_raw(Box::new(0_u8)).cast();
    }
    INSTANCES.with(|counts| {
        let (created, dropped) = counts.get();
        counts.set((created + 1, dropped));
    });
    ffi::SQLITE_OK
}

unsafe extern "C" fn delete_probe(instance: *mut ffi::Fts5Tokenizer) {
    unsafe {
        drop(Box::from_raw(instance.cast::<u8>()));
    }
    INSTANCES.with(|counts| {
        let (created, dropped) = counts.get();
        counts.set((created, dropped + 1));
    });
}

type TokenCallback =
    unsafe extern "C" fn(*mut c_void, c_int, *const c_char, c_int, c_int, c_int) -> c_int;

unsafe extern "C" fn tokenize_probe(
    _: *mut ffi::Fts5Tokenizer,
    data: *mut c_void,
    flags: c_int,
    _: *const c_char,
    _: c_int,
    callback: Option<TokenCallback>,
) -> c_int {
    if flags != ffi::FTS5_TOKENIZE_DOCUMENT {
        return ffi::SQLITE_MISUSE;
    }
    let Some(callback) = callback else {
        return ffi::SQLITE_MISUSE;
    };
    unsafe { callback(data, 0, c"x".as_ptr(), 1, 0, 1) }
}

#[test]
fn document_instance_is_deleted_on_success_and_callback_failure() {
    let connection = Connection::open_in_memory().unwrap();
    let mut tokenizer = ffi::fts5_tokenizer {
        xCreate: Some(create_probe),
        xDelete: Some(delete_probe),
        xTokenize: Some(tokenize_probe),
    };
    with_registration_api(&connection, |api| unsafe {
        let create = required((*api).xCreateTokenizer, "missing tokenizer registration")?;
        check(
            create(
                api,
                c"lode_document_probe".as_ptr(),
                ptr::null_mut(),
                &raw mut tokenizer,
                None,
            ),
            "probe registration failed",
        )
    })
    .unwrap();
    INSTANCES.with(|counts| counts.set((0, 0)));
    assert_eq!(
        builtin_document_tokens(&connection, c"lode_document_probe", "x")
            .unwrap()
            .len(),
        1
    );
    assert!(builtin_document_tokens(&connection, c"lode_document_probe", "中").is_err());
    assert_eq!(INSTANCES.with(std::cell::Cell::get), (2, 2));
}
