//! Experimental FTS5 auxiliary function; deliberately outside the core API.

use std::ffi::{c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use rusqlite::{Connection, ffi};
use serde::{Deserialize, Serialize};

const K1: f64 = 1.2;
const B: f64 = 0.75;
const IDF_FLOOR: f64 = 1e-6;

type SqlResult<T> = Result<T, c_int>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Hit {
    pub column: c_int,
    pub token_offset: c_int,
    // SQLite 3.46 cannot expose document tokens for prefix hits.
    pub document_tokens: Vec<Option<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Phrase {
    pub index: c_int,
    pub query_tokens: Vec<String>,
    pub document_frequency: i64,
    pub idf: f64,
    pub weighted_frequency: f64,
    pub bm25_contribution: f64,
    pub hits: Vec<Hit>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Explanation {
    pub rowid: i64,
    pub extension_api_version: c_int,
    pub corpus_rows: i64,
    pub average_length: f64,
    pub document_length: c_int,
    pub phrases: Vec<Phrase>,
}

struct Statistics {
    corpus_rows: i64,
    average_length: f64,
    frequencies: Vec<i64>,
    idfs: Vec<f64>,
}

fn check(code: c_int) -> SqlResult<()> {
    if code == ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(code)
    }
}

/// Registers on an owned, otherwise idle connection. No user-data pointers
/// survive registration; FTS5 owns and destroys each query's statistics cache.
pub fn register(connection: &Connection) -> rusqlite::Result<()> {
    let code = unsafe {
        let mut api: *mut ffi::fts5_api = ptr::null_mut();
        let mut statement = ptr::null_mut();
        let db = connection.handle();
        check(ffi::sqlite3_prepare_v2(
            db,
            c"SELECT fts5(?1)".as_ptr(),
            -1,
            &raw mut statement,
            ptr::null_mut(),
        ))
        .and_then(|()| {
            let bound = ffi::sqlite3_bind_pointer(
                statement,
                1,
                (&raw mut api).cast(),
                c"fts5_api_ptr".as_ptr(),
                None,
            );
            let stepped = if bound == ffi::SQLITE_OK {
                ffi::sqlite3_step(statement)
            } else {
                bound
            };
            let finalized = ffi::sqlite3_finalize(statement);
            check(bound)?;
            if stepped != ffi::SQLITE_ROW {
                return Err(stepped);
            }
            check(finalized)?;
            let api = api.as_ref().ok_or(ffi::SQLITE_MISUSE)?;
            if api.iVersion < 2 {
                return Err(ffi::SQLITE_MISUSE);
            }
            check(api.xCreateFunction.ok_or(ffi::SQLITE_MISUSE)?(
                ptr::from_ref(api).cast_mut(),
                c"phrase_probe".as_ptr(),
                ptr::null_mut(),
                Some(explain),
                None,
            ))
        })
    };
    code.map_err(|code| rusqlite::Error::SqliteFailure(ffi::Error::new(code), None))
}

unsafe extern "C" fn count_row(
    _: *const ffi::Fts5ExtensionApi,
    _: *mut ffi::Fts5Context,
    data: *mut c_void,
) -> c_int {
    // xQueryPhrase invokes this synchronously with our live stack counter.
    unsafe {
        *data.cast::<i64>() += 1;
    }
    ffi::SQLITE_OK
}

unsafe extern "C" fn drop_statistics(data: *mut c_void) {
    // xSetAuxdata transfers exactly one Box to FTS5, even on API failure.
    unsafe {
        drop(Box::from_raw(data.cast::<Statistics>()));
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Mirror SQLite's i64-to-double BM25 corpus statistics"
)]
unsafe fn statistics(
    api: &ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
) -> SqlResult<*const Statistics> {
    unsafe {
        let cached = api.xGetAuxdata.ok_or(ffi::SQLITE_MISUSE)?(context, 0).cast::<Statistics>();
        if !cached.is_null() {
            return Ok(cached);
        }
        let mut corpus_rows = 0;
        let mut total_tokens = 0;
        check(api.xRowCount.ok_or(ffi::SQLITE_MISUSE)?(
            context,
            &raw mut corpus_rows,
        ))?;
        check(api.xColumnTotalSize.ok_or(ffi::SQLITE_MISUSE)?(
            context,
            -1,
            &raw mut total_tokens,
        ))?;
        if corpus_rows <= 0 || total_tokens <= 0 {
            return Err(ffi::SQLITE_MISUSE);
        }
        let mut stats = Box::new(Statistics {
            corpus_rows,
            average_length: total_tokens as f64 / corpus_rows as f64,
            frequencies: Vec::new(),
            idfs: Vec::new(),
        });
        for phrase in 0..api.xPhraseCount.ok_or(ffi::SQLITE_MISUSE)?(context) {
            let mut count = 0_i64;
            check(api.xQueryPhrase.ok_or(ffi::SQLITE_MISUSE)?(
                context,
                phrase,
                (&raw mut count).cast(),
                Some(count_row),
            ))?;
            let idf = ((corpus_rows as f64 - count as f64 + 0.5) / (count as f64 + 0.5)).ln();
            stats.frequencies.push(count);
            stats.idfs.push(if idf <= 0.0 { IDF_FLOOR } else { idf });
        }
        let set_auxdata = api.xSetAuxdata.ok_or(ffi::SQLITE_MISUSE)?;
        let data = Box::into_raw(stats);
        check(set_auxdata(context, data.cast(), Some(drop_statistics)))?;
        Ok(data)
    }
}

unsafe fn token_text(token: *const c_char, length: c_int) -> SqlResult<Option<String>> {
    if token.is_null() {
        return Ok(None);
    }
    let length = usize::try_from(length).map_err(|_| ffi::SQLITE_MISUSE)?;
    let bytes = unsafe { std::slice::from_raw_parts(token.cast::<u8>(), length) };
    String::from_utf8(bytes.to_vec())
        .map(Some)
        .map_err(|_| ffi::SQLITE_ERROR)
}

unsafe fn collect(
    api: &ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
    weights: &[f64],
) -> SqlResult<Explanation> {
    // FTS5 lends the API and context for this invocation; cached statistics
    // remain alive until the MATCH statement finishes. No pointers escape.
    unsafe {
        if api.iVersion < 3 {
            return Err(ffi::SQLITE_MISUSE);
        }
        let stats = &*statistics(api, context)?;
        let mut document_length = 0;
        check(api.xColumnSize.ok_or(ffi::SQLITE_MISUSE)?(
            context,
            -1,
            &raw mut document_length,
        ))?;
        let mut phrases = Vec::new();
        for index in 0..api.xPhraseCount.ok_or(ffi::SQLITE_MISUSE)?(context) {
            let i = usize::try_from(index).map_err(|_| ffi::SQLITE_MISUSE)?;
            let mut query_tokens = Vec::new();
            for token_index in 0..api.xPhraseSize.ok_or(ffi::SQLITE_MISUSE)?(context, index) {
                let mut text = ptr::null();
                let mut length = 0;
                check(api.xQueryToken.ok_or(ffi::SQLITE_MISUSE)?(
                    context,
                    index,
                    token_index,
                    &raw mut text,
                    &raw mut length,
                ))?;
                query_tokens.push(token_text(text, length)?.ok_or(ffi::SQLITE_MISUSE)?);
            }
            phrases.push(Phrase {
                index,
                query_tokens,
                document_frequency: stats.frequencies[i],
                idf: stats.idfs[i],
                weighted_frequency: 0.0,
                bm25_contribution: 0.0,
                hits: Vec::new(),
            });
        }
        let mut instances = 0;
        check(api.xInstCount.ok_or(ffi::SQLITE_MISUSE)?(
            context,
            &raw mut instances,
        ))?;
        for instance in 0..instances {
            let (mut index, mut column, mut token_offset) = (0, 0, 0);
            check(api.xInst.ok_or(ffi::SQLITE_MISUSE)?(
                context,
                instance,
                &raw mut index,
                &raw mut column,
                &raw mut token_offset,
            ))?;
            let phrase = phrases
                .get_mut(usize::try_from(index).map_err(|_| ffi::SQLITE_MISUSE)?)
                .ok_or(ffi::SQLITE_MISUSE)?;
            let mut document_tokens = Vec::new();
            for token_index in 0..api.xPhraseSize.ok_or(ffi::SQLITE_MISUSE)?(context, index) {
                let mut text = ptr::null();
                let mut length = 0;
                check(api.xInstToken.ok_or(ffi::SQLITE_MISUSE)?(
                    context,
                    instance,
                    token_index,
                    &raw mut text,
                    &raw mut length,
                ))?;
                document_tokens.push(token_text(text, length)?);
            }
            phrase.weighted_frequency += weights
                .get(usize::try_from(column).map_err(|_| ffi::SQLITE_MISUSE)?)
                .copied()
                .unwrap_or(1.0);
            phrase.hits.push(Hit {
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
        }
        Ok(Explanation {
            rowid: api.xRowid.ok_or(ffi::SQLITE_MISUSE)?(context),
            extension_api_version: api.iVersion,
            corpus_rows: stats.corpus_rows,
            average_length: stats.average_length,
            document_length,
            phrases,
        })
    }
}

unsafe extern "C" fn explain(
    api: *const ffi::Fts5ExtensionApi,
    context: *mut ffi::Fts5Context,
    result: *mut ffi::sqlite3_context,
    count: c_int,
    values: *mut *mut ffi::sqlite3_value,
) {
    // Panics must never unwind into SQLite's C stack.
    let outcome = catch_unwind(AssertUnwindSafe(|| unsafe {
        let api = api.as_ref().ok_or(ffi::SQLITE_MISUSE)?;
        let count = usize::try_from(count).map_err(|_| ffi::SQLITE_MISUSE)?;
        let weights = if count == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(values, count)
                .iter()
                .map(|value| ffi::sqlite3_value_double(*value))
                .collect()
        };
        let explanation = collect(api, context, &weights)?;
        serde_json::to_string(&explanation).map_err(|_| ffi::SQLITE_ERROR)
    }));
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
            Ok(Err(code)) => ffi::sqlite3_result_error_code(result, code),
            Err(_) => ffi::sqlite3_result_error_code(result, ffi::SQLITE_ERROR),
        }
    }
}
