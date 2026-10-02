#![warn(clippy::pedantic)]

//! Built-in DOCUMENT callbacks; AUX tokenization has different synonym flags.

use lode_simple_native::Token;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use rusqlite::{Connection, ffi};

use super::{Fault, check, nonnegative, required, with_registration_api};

struct TokenizerInstance {
    pointer: *mut ffi::Fts5Tokenizer,
    delete: unsafe extern "C" fn(*mut ffi::Fts5Tokenizer),
}

impl Drop for TokenizerInstance {
    fn drop(&mut self) {
        // xCreate transferred this instance; xDelete runs on every exit.
        unsafe { (self.delete)(self.pointer) };
    }
}

pub(crate) fn builtin_document_tokens(
    connection: &Connection,
    tokenizer: &CStr,
    text: &str,
) -> rusqlite::Result<Vec<Token>> {
    with_registration_api(connection, |api| unsafe {
        let find = required((*api).xFindTokenizer, "missing FTS5 xFindTokenizer")?;
        let mut user_data = ptr::null_mut();
        let mut methods: ffi::fts5_tokenizer = std::mem::zeroed();
        check(
            find(
                api,
                tokenizer.as_ptr(),
                &raw mut user_data,
                &raw mut methods,
            ),
            "xFindTokenizer failed",
        )?;
        let create = required(methods.xCreate, "missing tokenizer xCreate")?;
        let delete = required(methods.xDelete, "missing tokenizer xDelete")?;
        let tokenize = required(methods.xTokenize, "missing tokenizer xTokenize")?;
        let length =
            c_int::try_from(text.len()).map_err(|_| Fault::invalid("DOCUMENT text too long"))?;
        let mut pointer = ptr::null_mut();
        // The Store verifies the fixed table definition, including the absence
        // of tokenizer arguments, before this default instance is used.
        check(
            create(user_data, ptr::null_mut(), 0, &raw mut pointer),
            "tokenizer xCreate failed",
        )?;
        if pointer.is_null() {
            return Err(Fault::invalid("tokenizer xCreate returned no instance"));
        }
        let instance = TokenizerInstance { pointer, delete };
        let mut collector = Collector {
            input: text,
            tokens: Vec::new(),
            error: None,
        };
        // SQLite 3.46's trigram tokenizer reads the terminating NUL even
        // though nText is supplied. Own that byte without rejecting embedded NUL.
        let mut terminated = text.as_bytes().to_vec();
        terminated.push(0);
        let code = tokenize(
            instance.pointer,
            (&raw mut collector).cast(),
            ffi::FTS5_TOKENIZE_DOCUMENT,
            terminated.as_ptr().cast(),
            length,
            Some(collect_token),
        );
        if let Some(error) = collector.error {
            return Err(error);
        }
        check(code, "DOCUMENT tokenization failed")?;
        Ok(collector.tokens)
    })
}

struct Collector<'a> {
    input: &'a str,
    tokens: Vec<Token>,
    error: Option<Fault>,
}

unsafe extern "C" fn collect_token(
    data: *mut c_void,
    flags: c_int,
    token: *const c_char,
    length: c_int,
    start: c_int,
    end: c_int,
) -> c_int {
    // FTS5 calls synchronously. The borrowed collector and token bytes cannot
    // outlive this call; Rust panics must not unwind through SQLite.
    let Some(collector) = (unsafe { data.cast::<Collector<'_>>().as_mut() }) else {
        return ffi::SQLITE_MISUSE;
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let flags = u32::try_from(flags).map_err(|_| Fault::invalid("invalid DOCUMENT flags"))?;
        let (length, start, end) = (nonnegative(length)?, nonnegative(start)?, nonnegative(end)?);
        if token.is_null() || start >= end || collector.input.get(start..end).is_none() {
            return Err(Fault::invalid("invalid DOCUMENT token bounds"));
        }
        let bytes = unsafe { std::slice::from_raw_parts(token.cast::<u8>(), length) };
        let normalized = std::str::from_utf8(bytes)
            .map_err(|_| Fault::invalid("invalid DOCUMENT token UTF-8"))?;
        if normalized.is_empty() {
            return Err(Fault::invalid("empty DOCUMENT token"));
        }
        collector.tokens.push(Token {
            flags,
            text: normalized.into(),
            start,
            end,
        });
        Ok::<_, Fault>(())
    }));
    match result {
        Ok(Ok(())) => ffi::SQLITE_OK,
        Ok(Err(error)) => {
            collector.error = Some(error);
            error.code
        }
        Err(_) => {
            collector.error = Some(Fault::invalid("DOCUMENT callback panic"));
            ffi::SQLITE_ERROR
        }
    }
}

#[cfg(test)]
mod tests;
