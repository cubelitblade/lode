use std::cell::Cell;

use rusqlite::functions::FunctionFlags;

use super::*;

fn fixture(strategy: &str) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    if matches!(strategy, "simple" | "jieba") {
        lode_simple_native::register_connection(&conn, strategy == "jieba").unwrap();
    }
    let physical = if strategy == "jieba" {
        "simple"
    } else {
        strategy
    };
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE documents USING fts5(text, title, tokenize='{physical}');"
    ))
    .unwrap();
    for rowid in 1..=31 {
        let text = if rowid <= 11 {
            "alpha beta alpha 知识 common"
        } else {
            "filler unrelated common"
        };
        let title = if rowid % 2 == 1 {
            "alpha 知识"
        } else {
            "beta 中国"
        };
        conn.execute(
            "INSERT INTO documents(rowid, text, title) VALUES (?1, ?2, ?3)",
            rusqlite::params![rowid, text, title],
        )
        .unwrap();
    }
    register(&conn).unwrap();
    conn
}

fn counts() -> (usize, usize) {
    CACHE_COUNTS.with(Cell::get)
}

fn reset_counts() {
    let (created, dropped) = counts();
    assert_eq!(created, dropped, "previous test leaked statistics");
    CACHE_COUNTS.with(|counts| counts.set((0, 0)));
}

fn assert_released() {
    let (created, dropped) = counts();
    assert!(created > 0);
    assert_eq!(created, dropped, "statistics not released");
}

fn details(conn: &Connection, expression: &str) -> RowDetails {
    let json: String = conn.query_row(
        "SELECT lode_fts_details(documents) FROM documents WHERE documents MATCH ?1 AND rowid=1",
        [expression], |row| row.get(0),
    ).unwrap();
    serde_json::from_str(&json).unwrap()
}

#[test]
fn weights_column_filters_and_zero_weights_match_native_bm25() {
    crate::run_with_isolated_jieba_cache(
        "fts::auxiliary::tests::weights_column_filters_and_zero_weights_match_native_bm25",
        || {
            for strategy in ["unicode61", "trigram", "simple", "jieba"] {
                let conn = fixture(strategy);
                for (expression, weights) in [
                    ("\"alpha\" OR \"beta\"", ", 3.0, 0.25"),
                    ("title : \"alpha\"", ", 3.0, 0.25"),
                    ("\"alpha\"", ", 0.0, 0.0"),
                ] {
                    let sql = format!(
                        "SELECT rowid, bm25(documents{weights}), lode_fts_details(documents{weights})
                         FROM documents WHERE documents MATCH ?1 ORDER BY rowid"
                    );
                    let mut statement = conn.prepare(&sql).unwrap();
                    let evidence: Vec<(i64, f64, String)> = statement
                        .query_map([expression], |row| {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                        })
                        .unwrap()
                        .collect::<rusqlite::Result<_>>()
                        .unwrap();
                    assert!(!evidence.is_empty());
                    for (rowid, score, json) in evidence {
                        let row: RowDetails = serde_json::from_str(&json).unwrap();
                        let matched = row.into_match(rowid, score).unwrap();
                        assert_eq!(matched.corpus_rows, 31);
                        if expression.starts_with("title") {
                            assert_eq!(matched.phrases[0].document_frequency, 16);
                            assert!(matched.phrases[0].hits.iter().all(|hit| hit.column == 1));
                        }
                        if weights.contains("0.0") {
                            assert!(matched.bm25.abs() < f64::EPSILON);
                            assert!(
                                matched
                                    .phrases
                                    .iter()
                                    .all(|phrase| phrase.bm25_contribution.abs() < f64::EPSILON)
                            );
                        }
                        let filtered_sql = format!(
                            "SELECT bm25(documents{weights}), lode_fts_details(documents{weights})
                             FROM documents WHERE documents MATCH ?1 AND rowid=?2"
                        );
                        let (filtered_score, filtered_json): (f64, String) = conn
                            .query_row(&filtered_sql, rusqlite::params![expression, rowid], |row| {
                                Ok((row.get(0)?, row.get(1)?))
                            })
                            .unwrap();
                        let filtered: RowDetails = serde_json::from_str(&filtered_json).unwrap();
                        assert_eq!(filtered.into_match(rowid, filtered_score).unwrap(), matched);
                    }
                }
            }
        },
    );
}

#[test]
fn query_statistics_are_cached_per_cursor_and_released_on_early_stop() {
    reset_counts();
    let conn = fixture("unicode61");
    {
        let mut statement = conn
            .prepare(
                "SELECT lode_fts_details(documents), lode_fts_details(documents)
             FROM documents WHERE documents MATCH ?1",
            )
            .unwrap();
        {
            let mut rows = statement.query(["alpha"]).unwrap();
            assert!(rows.next().unwrap().is_some());
            assert_eq!(
                counts(),
                (1, 0),
                "one cache shared by auxiliary invocations"
            );
            assert!(rows.next().unwrap().is_some());
            assert_eq!(counts(), (1, 0), "cache reused across target rows");
        }
        // Dropping Rows resets the statement, even when matches remain unread.
        assert_released();
        {
            let mut rows = statement.query(["beta"]).unwrap();
            assert!(rows.next().unwrap().is_some());
        }
        assert_eq!(counts(), (2, 2));
    }
    drop(conn);
    assert_released();
}

#[test]
fn reused_statements_do_not_keep_old_phrase_frequencies() {
    reset_counts();
    let conn = fixture("unicode61");
    let mut statement = conn.prepare(
        "SELECT lode_fts_details(documents) FROM documents WHERE documents MATCH ?1 AND rowid=1",
    ).unwrap();
    let read = |statement: &mut rusqlite::Statement<'_>, expression: &str| {
        let json: String = statement.query_row([expression], |row| row.get(0)).unwrap();
        serde_json::from_str::<RowDetails>(&json).unwrap()
    };
    let alpha = read(&mut statement, "alpha");
    let common = read(&mut statement, "common");
    assert_eq!(alpha.phrases[0].document_frequency, 21);
    assert_eq!(common.phrases[0].document_frequency, 31);
    conn.execute("INSERT INTO documents VALUES ('alpha', 'alpha')", [])
        .unwrap();
    let updated = read(&mut statement, "alpha");
    assert_eq!(updated.corpus_rows, 32);
    assert_eq!(updated.phrases[0].document_frequency, 22);
    assert_released();
}

#[test]
fn token_copy_rejects_invalid_buffers_and_preserves_prefix_absence() {
    let invalid_utf8 = [0xff_u8];
    let valid_utf8 = "矿".as_bytes();
    unsafe {
        assert!(token_text(invalid_utf8.as_ptr().cast(), 1).is_err());
        assert!(token_text(valid_utf8.as_ptr().cast(), -1).is_err());
        assert!(token_text(ptr::null(), 1).is_err());
        assert_eq!(token_text(ptr::null(), 0).unwrap(), None);
        assert_eq!(
            token_text(valid_utf8.as_ptr().cast(), 3).unwrap(),
            Some("矿".to_string())
        );
    }
}

#[test]
fn count_callback_rejects_null_data_and_overflow() {
    unsafe {
        assert_eq!(
            count_row(ptr::null(), ptr::null_mut(), ptr::null_mut()),
            ffi::SQLITE_MISUSE
        );
        let mut count = i64::MAX;
        assert_eq!(
            count_row(ptr::null(), ptr::null_mut(), (&raw mut count).cast()),
            ffi::SQLITE_TOOBIG
        );
        assert_eq!(count, i64::MAX);
    }
}

#[derive(Clone, Copy)]
enum Inject {
    RowCount,
    QueryPhrase,
    SetAuxdata,
    Instances,
    MissingTokenApi,
    OldApi,
    Panic,
    Interrupt,
}

thread_local! {
    static INJECT: Cell<Inject> = const { Cell::new(Inject::RowCount) };
}

unsafe extern "C" fn failed_row_count(_: *mut ffi::Fts5Context, _: *mut i64) -> c_int {
    ffi::SQLITE_IOERR
}

unsafe extern "C" fn failed_query_phrase(
    _: *mut ffi::Fts5Context,
    _: c_int,
    _: *mut c_void,
    _: Option<
        unsafe extern "C" fn(
            *const ffi::Fts5ExtensionApi,
            *mut ffi::Fts5Context,
            *mut c_void,
        ) -> c_int,
    >,
) -> c_int {
    ffi::SQLITE_IOERR
}

unsafe extern "C" fn failed_set_auxdata(
    _: *mut ffi::Fts5Context,
    data: *mut c_void,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> c_int {
    // Emulate the documented FTS5 OOM path: ownership was already transferred.
    if let Some(destroy) = destroy {
        unsafe { destroy(data) };
    }
    ffi::SQLITE_NOMEM
}

unsafe extern "C" fn failed_instances(_: *mut ffi::Fts5Context, _: *mut c_int) -> c_int {
    ffi::SQLITE_IOERR
}

unsafe extern "C" fn injected(
    api: *const ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
    result: *mut ffi::sqlite3_context,
    count: c_int,
    values: *mut *mut ffi::sqlite3_value,
) {
    unsafe {
        match INJECT.with(Cell::get) {
            Inject::OldApi => {
                // Only the ABI prefix exists; the version check must run
                // before a full Fts5ExtensionApi reference is constructed.
                let version: c_int = 2;
                explain(
                    ptr::from_ref(&version).cast(),
                    context,
                    result,
                    count,
                    values,
                );
            }
            Inject::Panic => return_result(result, || {
                statistics(&*api, context)?;
                panic!("intentional Rust panic after cache ownership transfer")
            }),
            Inject::Interrupt => return_result(result, || {
                statistics(&*api, context)?;
                Err(Fault {
                    code: ffi::SQLITE_INTERRUPT,
                    message: "interrupted FTS diagnostic",
                })
            }),
            mode => {
                let mut modified = *api;
                match mode {
                    Inject::RowCount => modified.xRowCount = Some(failed_row_count),
                    Inject::QueryPhrase => modified.xQueryPhrase = Some(failed_query_phrase),
                    Inject::SetAuxdata => modified.xSetAuxdata = Some(failed_set_auxdata),
                    Inject::Instances => modified.xInstCount = Some(failed_instances),
                    Inject::MissingTokenApi => modified.xQueryToken = None,
                    _ => {}
                }
                explain(&raw const modified, context, result, count, values);
            }
        }
    }
}

fn fail_query(conn: &Connection, mode: Inject) -> rusqlite::Error {
    INJECT.with(|inject| inject.set(mode));
    register_function(conn, c"fault", Some(injected)).unwrap();
    conn.query_row(
        "SELECT fault(documents) FROM documents WHERE documents MATCH 'alpha'",
        [],
        |row| row.get::<_, String>(0),
    )
    .unwrap_err()
}

#[test]
fn api_failures_and_panics_do_not_leak_or_unwind_into_sqlite() {
    for mode in [
        Inject::RowCount,
        Inject::QueryPhrase,
        Inject::SetAuxdata,
        Inject::Instances,
        Inject::MissingTokenApi,
        Inject::OldApi,
        Inject::Panic,
        Inject::Interrupt,
    ] {
        reset_counts();
        let conn = fixture("unicode61");
        let error = fail_query(&conn, mode);
        let expected = match mode {
            Inject::SetAuxdata => ffi::SQLITE_NOMEM,
            Inject::RowCount | Inject::QueryPhrase | Inject::Instances => ffi::SQLITE_IOERR,
            Inject::MissingTokenApi => ffi::SQLITE_MISUSE,
            Inject::Interrupt => ffi::SQLITE_INTERRUPT,
            Inject::OldApi | Inject::Panic => ffi::SQLITE_ERROR,
        };
        let rusqlite::Error::SqliteFailure(sqlite, message) = error else {
            panic!("{error:?}")
        };
        assert_eq!(sqlite.extended_code, expected);
        assert!(message.is_some());
        if matches!(mode, Inject::Panic) {
            assert!(message.unwrap().contains("callback panicked"));
        }
        let (created, dropped) = counts();
        assert_eq!(created, dropped, "cache leaked after callback failure");
        if matches!(
            mode,
            Inject::SetAuxdata
                | Inject::Instances
                | Inject::MissingTokenApi
                | Inject::Panic
                | Inject::Interrupt
        ) {
            assert_eq!(created, 1);
        }
        // A failed explicit diagnostic must leave the connection reusable.
        assert_eq!(details(&conn, "beta").phrases[0].query_tokens, ["beta"]);
        drop(conn);
        assert_released();
    }
}

#[test]
fn registration_finalizes_its_statement_on_success_and_failure() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    assert!(unsafe { ffi::sqlite3_next_stmt(conn.handle(), ptr::null_mut()) }.is_null());
    conn.create_scalar_function("fts5", 1, FunctionFlags::SQLITE_UTF8, |_| Ok(0))
        .unwrap();
    let error = register(&conn).unwrap_err();
    assert!(error.to_string().contains("registration API version"));
    assert!(unsafe { ffi::sqlite3_next_stmt(conn.handle(), ptr::null_mut()) }.is_null());
    assert_eq!(
        conn.query_row("SELECT 42", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        42
    );
    conn.create_scalar_function(
        "fts5",
        1,
        FunctionFlags::SQLITE_UTF8,
        |_| -> rusqlite::Result<i64> {
            Err(rusqlite::Error::UserFunctionError(Box::new(
                std::io::Error::other("forced FTS5 API lookup failure"),
            )))
        },
    )
    .unwrap();
    let error = register(&conn).unwrap_err();
    assert!(error.to_string().contains("forced FTS5 API lookup failure"));
    assert!(unsafe { ffi::sqlite3_next_stmt(conn.handle(), ptr::null_mut()) }.is_null());
}

#[test]
fn nonfinite_weights_are_errors_and_leave_no_cache() {
    reset_counts();
    let conn = fixture("unicode61");
    let error = conn
        .query_row(
            "SELECT lode_fts_details(documents, 1e999)
         FROM documents WHERE documents MATCH 'alpha'",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_err();
    assert!(error.to_string().contains("nonfinite"));
    assert_eq!(counts(), (0, 0));
}
