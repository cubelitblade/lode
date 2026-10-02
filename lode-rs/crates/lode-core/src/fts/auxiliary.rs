#![warn(clippy::pedantic)]

//! Private FTS5 auxiliary boundary. SQLite lends callback pointers; only owned
//! data leaves a callback. Query statistics are owned and destroyed by FTS5.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use rusqlite::{Connection, ffi};

use super::explanation::{FtsHit, FtsPhrase, RowDetails};

// Keep these aligned with the bundled SQLite fts5Bm25Function.
const K1: f64 = 1.2;
const B: f64 = 0.75;
const IDF_FLOOR: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct Fault {
    code: c_int,
    message: &'static str,
}

impl Fault {
    fn invalid(message: &'static str) -> Self {
        Self {
            code: ffi::SQLITE_ERROR,
            message,
        }
    }
}

type SqlResult<T> = Result<T, Fault>;

fn check(code: c_int, operation: &'static str) -> SqlResult<()> {
    if code == ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(Fault {
            code,
            message: operation,
        })
    }
}

fn required<T>(value: Option<T>, name: &'static str) -> SqlResult<T> {
    value.ok_or(Fault {
        code: ffi::SQLITE_MISUSE,
        message: name,
    })
}

fn nonnegative(value: c_int) -> SqlResult<usize> {
    usize::try_from(value).map_err(|_| Fault::invalid("negative FTS5 size or index"))
}

struct RawStatement(*mut ffi::sqlite3_stmt);

impl Drop for RawStatement {
    fn drop(&mut self) {
        // The guard owns this statement, including any partially prepared one.
        unsafe { ffi::sqlite3_finalize(self.0) };
    }
}

/// Register only for an explicitly requested diagnostic, on an idle connection.
pub(crate) fn register(connection: &Connection) -> rusqlite::Result<()> {
    register_function(connection, c"lode_fts_details", Some(explain))
}

fn register_function(
    connection: &Connection,
    name: &CStr,
    callback: ffi::fts5_extension_function,
) -> rusqlite::Result<()> {
    // The owned connection stays live; the prepared statement retains the
    // stack output pointer only until it is finalized before this call returns.
    unsafe {
        let db = connection.handle();
        let mut api: *mut ffi::fts5_api = ptr::null_mut();
        let mut statement = ptr::null_mut();
        let prepared = ffi::sqlite3_prepare_v2(
            db,
            c"SELECT fts5(?1)".as_ptr(),
            -1,
            &raw mut statement,
            ptr::null_mut(),
        );
        let guard = RawStatement(statement);
        let result = (|| {
            check(prepared, "sqlite3_prepare_v2 failed")?;
            check(
                ffi::sqlite3_bind_pointer(
                    guard.0,
                    1,
                    (&raw mut api).cast(),
                    c"fts5_api_ptr".as_ptr(),
                    None,
                ),
                "sqlite3_bind_pointer failed",
            )?;
            let stepped = ffi::sqlite3_step(guard.0);
            if stepped != ffi::SQLITE_ROW {
                return Err(Fault {
                    code: stepped,
                    message: "could not obtain the FTS5 API",
                });
            }
            if api.is_null() || api.cast::<c_int>().read() < 2 {
                return Err(Fault::invalid(
                    "FTS5 registration API version 2 is required",
                ));
            }
            let create = required((*api).xCreateFunction, "missing FTS5 xCreateFunction")?;
            check(
                create(api, name.as_ptr(), ptr::null_mut(), callback, None),
                "xCreateFunction failed",
            )
        })();
        // Preserve SQLite's message before finalization can change it.
        let message = CStr::from_ptr(ffi::sqlite3_errmsg(db))
            .to_string_lossy()
            .into_owned();
        drop(guard);
        result.map_err(|fault| {
            rusqlite::Error::SqliteFailure(
                ffi::Error::new(fault.code),
                Some(format!("{}: {message}", fault.message)),
            )
        })
    }
}

struct Statistics {
    corpus_rows: i64,
    average_length: f64,
    frequencies: Vec<i64>,
    idfs: Vec<f64>,
}

#[cfg(test)]
thread_local! {
    // FTS callbacks are synchronous on the executing test thread.
    static CACHE_COUNTS: std::cell::Cell<(usize, usize)> = const {
        std::cell::Cell::new((0, 0))
    };
}

#[cfg(test)]
impl Drop for Statistics {
    fn drop(&mut self) {
        CACHE_COUNTS.with(|counts| {
            let (created, dropped) = counts.get();
            counts.set((created, dropped + 1));
        });
    }
}

unsafe extern "C" fn count_row(
    _: *const ffi::Fts5ExtensionApi,
    _: *mut ffi::Fts5Context,
    data: *mut c_void,
) -> c_int {
    // xQueryPhrase invokes this synchronously with a borrowed stack counter.
    let Some(count) = (unsafe { data.cast::<i64>().as_mut() }) else {
        return ffi::SQLITE_MISUSE;
    };
    let Some(next) = count.checked_add(1) else {
        return ffi::SQLITE_TOOBIG;
    };
    *count = next;
    ffi::SQLITE_OK
}

unsafe extern "C" fn drop_statistics(data: *mut c_void) {
    if !data.is_null() {
        // xSetAuxdata owns exactly one Box, and also destroys it on failure.
        unsafe { drop(Box::from_raw(data.cast::<Statistics>())) };
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Mirror SQLite's i64-to-double BM25 statistics"
)]
unsafe fn statistics(
    api: &ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
) -> SqlResult<*const Statistics> {
    unsafe {
        let get = required(api.xGetAuxdata, "missing FTS5 xGetAuxdata")?;
        let cached = get(context, 0).cast::<Statistics>();
        if !cached.is_null() {
            return Ok(cached);
        }
        let mut corpus_rows = 0;
        let mut total_tokens = 0;
        check(
            required(api.xRowCount, "missing FTS5 xRowCount")?(context, &raw mut corpus_rows),
            "xRowCount failed",
        )?;
        check(
            required(api.xColumnTotalSize, "missing FTS5 xColumnTotalSize")?(
                context,
                -1,
                &raw mut total_tokens,
            ),
            "xColumnTotalSize failed",
        )?;
        if corpus_rows <= 0 || total_tokens <= 0 {
            return Err(Fault::invalid("invalid FTS5 corpus size"));
        }
        let phrase_count = required(api.xPhraseCount, "missing FTS5 xPhraseCount")?(context);
        nonnegative(phrase_count)?;
        let query_phrase = required(api.xQueryPhrase, "missing FTS5 xQueryPhrase")?;
        let mut frequencies = Vec::new();
        let mut idfs = Vec::new();
        for phrase in 0..phrase_count {
            let mut count = 0_i64;
            check(
                query_phrase(context, phrase, (&raw mut count).cast(), Some(count_row)),
                "xQueryPhrase failed",
            )?;
            if count < 0 || count > corpus_rows {
                return Err(Fault::invalid("invalid FTS5 phrase document frequency"));
            }
            let idf = ((corpus_rows as f64 - count as f64 + 0.5) / (count as f64 + 0.5)).ln();
            frequencies.push(count);
            idfs.push(if idf <= 0.0 { IDF_FLOOR } else { idf });
        }
        let set = required(api.xSetAuxdata, "missing FTS5 xSetAuxdata")?;
        let stats = Box::new(Statistics {
            corpus_rows,
            average_length: total_tokens as f64 / corpus_rows as f64,
            frequencies,
            idfs,
        });
        #[cfg(test)]
        CACHE_COUNTS.with(|counts| {
            let (created, dropped) = counts.get();
            counts.set((created + 1, dropped));
        });
        let data = Box::into_raw(stats);
        // On error, FTS5 calls the destructor before returning. Do not reclaim
        // the Box here, and do not dereference this pointer until set succeeds.
        check(
            set(context, data.cast(), Some(drop_statistics)),
            "xSetAuxdata failed",
        )?;
        Ok(data)
    }
}

unsafe fn token_text(token: *const c_char, length: c_int) -> SqlResult<Option<String>> {
    let length = nonnegative(length)?;
    if token.is_null() {
        return if length == 0 {
            Ok(None)
        } else {
            Err(Fault::invalid("null FTS5 token with a nonzero length"))
        };
    }
    // SQLite supplies a live token buffer of the stated byte length. Copy it
    // before any subsequent API call can invalidate the borrowed buffer.
    let bytes = unsafe { std::slice::from_raw_parts(token.cast::<u8>(), length) };
    std::str::from_utf8(bytes)
        .map(|text| Some(text.to_owned()))
        .map_err(|_| Fault::invalid("FTS5 token is not valid UTF-8"))
}

unsafe fn query_phrases(
    api: &ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
    stats: &Statistics,
) -> SqlResult<Vec<FtsPhrase>> {
    unsafe {
        let phrase_count = required(api.xPhraseCount, "missing FTS5 xPhraseCount")?(context);
        if nonnegative(phrase_count)? != stats.frequencies.len() {
            return Err(Fault::invalid("FTS5 phrase cache size mismatch"));
        }
        let phrase_size = required(api.xPhraseSize, "missing FTS5 xPhraseSize")?;
        let query_token = required(api.xQueryToken, "missing FTS5 xQueryToken")?;
        let mut phrases = Vec::new();
        for index in 0..phrase_count {
            let i = nonnegative(index)?;
            let size = phrase_size(context, index);
            if size <= 0 {
                return Err(Fault::invalid("invalid FTS5 phrase size"));
            }
            let mut query_tokens = Vec::new();
            for token_index in 0..size {
                let mut text = ptr::null();
                let mut length = 0;
                check(
                    query_token(context, index, token_index, &raw mut text, &raw mut length),
                    "xQueryToken failed",
                )?;
                query_tokens.push(
                    token_text(text, length)?
                        .ok_or_else(|| Fault::invalid("missing FTS5 query token"))?,
                );
            }
            phrases.push(FtsPhrase {
                index,
                query_tokens,
                document_frequency: stats.frequencies[i],
                idf: stats.idfs[i],
                weighted_frequency: 0.0,
                bm25_contribution: 0.0,
                hits: Vec::new(),
            });
        }
        Ok(phrases)
    }
}

unsafe fn collect(
    api: &ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
    weights: &[f64],
) -> SqlResult<RowDetails> {
    // The API/context are borrowed for this invocation, and auxdata is live
    // until this MATCH cursor is reset or finalized. No borrowed data escapes.
    unsafe {
        let stats = &*statistics(api, context)?;
        let mut document_length = 0;
        check(
            required(api.xColumnSize, "missing FTS5 xColumnSize")?(
                context,
                -1,
                &raw mut document_length,
            ),
            "xColumnSize failed",
        )?;
        nonnegative(document_length)?;
        let column_count = required(api.xColumnCount, "missing FTS5 xColumnCount")?(context);
        nonnegative(column_count)?;
        let mut phrases = query_phrases(api, context, stats)?;
        let phrase_size = required(api.xPhraseSize, "missing FTS5 xPhraseSize")?;
        let mut instances = 0;
        check(
            required(api.xInstCount, "missing FTS5 xInstCount")?(context, &raw mut instances),
            "xInstCount failed",
        )?;
        nonnegative(instances)?;
        let inst = required(api.xInst, "missing FTS5 xInst")?;
        let inst_token = required(api.xInstToken, "missing FTS5 xInstToken")?;
        for instance in 0..instances {
            let (mut index, mut column, mut token_offset) = (0, 0, 0);
            check(
                inst(
                    context,
                    instance,
                    &raw mut index,
                    &raw mut column,
                    &raw mut token_offset,
                ),
                "xInst failed",
            )?;
            nonnegative(token_offset)?;
            if column < 0 || column >= column_count {
                return Err(Fault::invalid("invalid FTS5 hit column"));
            }
            let phrase = phrases
                .get_mut(nonnegative(index)?)
                .ok_or_else(|| Fault::invalid("invalid FTS5 hit phrase index"))?;
            let mut document_tokens = Vec::new();
            for token_index in 0..phrase_size(context, index) {
                let mut text = ptr::null();
                let mut length = 0;
                check(
                    inst_token(
                        context,
                        instance,
                        token_index,
                        &raw mut text,
                        &raw mut length,
                    ),
                    "xInstToken failed",
                )?;
                document_tokens.push(token_text(text, length)?);
            }
            phrase.weighted_frequency += weights.get(nonnegative(column)?).copied().unwrap_or(1.0);
            phrase.hits.push(FtsHit {
                column,
                token_offset,
                document_tokens,
            });
        }
        for phrase in &mut phrases {
            let frequency = phrase.weighted_frequency;
            phrase.bm25_contribution = -phrase.idf
                * ((frequency * (K1 + 1.0))
                    / (frequency
                        + K1 * (1.0 - B + B * f64::from(document_length) / stats.average_length)));
            if !frequency.is_finite() || !phrase.bm25_contribution.is_finite() {
                return Err(Fault::invalid("nonfinite FTS5 phrase score"));
            }
        }
        Ok(RowDetails {
            rowid: required(api.xRowid, "missing FTS5 xRowid")?(context),
            corpus_rows: stats.corpus_rows,
            average_length: stats.average_length,
            document_length,
            phrases,
        })
    }
}

/// The unwind guard encloses all Rust work; result pointers remain SQLite-owned.
unsafe fn return_result(
    result: *mut ffi::sqlite3_context,
    build: impl FnOnce() -> SqlResult<String>,
) {
    let outcome = catch_unwind(AssertUnwindSafe(build));
    unsafe {
        match outcome {
            Ok(Ok(json)) => match c_int::try_from(json.len()) {
                Ok(length) => ffi::sqlite3_result_text(
                    result,
                    json.as_ptr().cast(),
                    length,
                    ffi::SQLITE_TRANSIENT(),
                ),
                Err(_) => ffi::sqlite3_result_error_code(result, ffi::SQLITE_TOOBIG),
            },
            failure => {
                let fault = match failure {
                    Ok(Err(fault)) => fault,
                    _ => Fault::invalid("FTS5 explanation callback panicked"),
                };
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    reason = "All fault messages are short static strings"
                )]
                ffi::sqlite3_result_error(
                    result,
                    fault.message.as_ptr().cast(),
                    fault.message.len() as c_int,
                );
                ffi::sqlite3_result_error_code(result, fault.code);
            }
        }
    }
}

unsafe extern "C" fn explain(
    api: *const ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
    result: *mut ffi::sqlite3_context,
    count: c_int,
    values: *mut *mut ffi::sqlite3_value,
) {
    // SQLite guarantees a live result context; malformed API data still becomes
    // a SQLite error, and Rust panics cannot unwind through its C stack.
    unsafe {
        return_result(result, || {
            if api.is_null() || context.is_null() {
                return Err(Fault::invalid("missing FTS5 callback context"));
            }
            // Read only the ABI prefix until the token API version is checked.
            if api.cast::<c_int>().read() < 3 {
                return Err(Fault::invalid("FTS5 extension API version 3 is required"));
            }
            let api = &*api;
            let count = nonnegative(count)?;
            let weights = if count == 0 {
                Vec::new()
            } else {
                if values.is_null() {
                    return Err(Fault::invalid("missing FTS5 column weights"));
                }
                std::slice::from_raw_parts(values, count)
                    .iter()
                    .map(|value| {
                        if value.is_null() {
                            return Err(Fault::invalid("null FTS5 column weight"));
                        }
                        let weight = ffi::sqlite3_value_double(*value);
                        if !weight.is_finite() {
                            return Err(Fault::invalid("nonfinite FTS5 column weight"));
                        }
                        Ok(weight)
                    })
                    .collect::<SqlResult<Vec<_>>>()?
            };
            let details = collect(api, context, &weights)?;
            serde_json::to_string(&details)
                .map_err(|_| Fault::invalid("could not serialize FTS5 details"))
        });
    }
}

#[cfg(test)]
mod tests;
