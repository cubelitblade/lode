#![warn(clippy::pedantic)]

//! Rust-owned boundary for the vendored `simple` FTS5 tokenizer.
//!
//! The tokenizer implementation remains upstream C++ because its indexing and
//! query semantics are part of the compatibility contract. This crate owns
//! registration, callback collection, and the embedded cppjieba dictionary
//! cache; SQLite itself continues to come from rusqlite's bundled build.

use std::ffi::{CStr, CString};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rusqlite::Connection;
use thiserror::Error;

const SQLITE_OK: i32 = 0;
const SQLITE_ERROR: i32 = 1;
/// FTS5 flag indicating that a token occupies the previous token's position.
pub const TOKEN_COLOCATED: u32 = 1;
const DICTIONARY_VERSION: &str = "simple-0.7.1-cppjieba-194c144d";
const DICTIONARY_MARKER: &str = ".complete";
const DICTIONARY_FILES: &[(&str, &[u8])] = &[
    (
        "jieba.dict.utf8",
        include_bytes!("../vendor/dict/jieba.dict.utf8"),
    ),
    (
        "hmm_model.utf8",
        include_bytes!("../vendor/dict/hmm_model.utf8"),
    ),
    (
        "user.dict.utf8",
        include_bytes!("../vendor/dict/user.dict.utf8"),
    ),
    ("idf.utf8", include_bytes!("../vendor/dict/idf.utf8")),
    (
        "stop_words.utf8",
        include_bytes!("../vendor/dict/stop_words.utf8"),
    ),
];

static TEMP_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Errors raised while registering or using the native tokenizer.
#[derive(Debug, Error)]
pub enum Error {
    #[error("native tokenizer SQLite error ({code}): {message}")]
    Sqlite { code: i32, message: String },
    #[error("native tokenizer callback failed: {0}")]
    Callback(String),
    #[error("native tokenizer input is not valid UTF-8: {0}")]
    InvalidInput(String),
    #[error("native tokenizer cache is unavailable: {0}")]
    Cache(String),
    #[error("native tokenizer cache I/O failed: {0}")]
    Io(#[from] io::Error),
}

/// One token emitted by the FTS5 document-tokenization callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// FTS5 callback flags. `1` means colocated with the previous token.
    pub flags: u32,
    /// The normalized token text emitted by the tokenizer.
    pub text: String,
    /// Inclusive byte start in the original UTF-8 input.
    pub start: usize,
    /// Exclusive byte end in the original UTF-8 input.
    pub end: usize,
}

unsafe extern "C" {
    fn lode_simple_register(db: *mut rusqlite::ffi::sqlite3) -> i32;
    fn lode_simple_last_error() -> *const libc::c_char;
    fn lode_simple_set_jieba_dict(path: *const libc::c_char, path_len: libc::c_int) -> i32;
    fn lode_simple_tokenize(
        db: *mut rusqlite::ffi::sqlite3,
        text: *const libc::c_char,
        text_len: libc::c_int,
        ctx: *mut libc::c_void,
        callback: unsafe extern "C" fn(
            *mut libc::c_void,
            libc::c_int,
            *const libc::c_char,
            libc::c_int,
            libc::c_int,
            libc::c_int,
        ) -> libc::c_int,
    ) -> i32;
}

/// Register the native tokenizer and query helpers on one SQLite connection.
///
/// # Errors
///
/// Returns an error when SQLite rejects registration or when the jieba cache
/// cannot be materialized and configured.
pub fn register_connection(conn: &Connection, with_jieba: bool) -> Result<(), Error> {
    let handle = unsafe { conn.handle() };
    let rc = unsafe { lode_simple_register(handle) };
    if rc != SQLITE_OK {
        return Err(sqlite_error(handle, rc));
    }

    if with_jieba {
        let directory = ensure_jieba_dictionary()?;
        let path = path_to_cstring(&directory)?;
        let path_len = i32::try_from(path.as_bytes().len())
            .map_err(|_| Error::Cache("dictionary cache path is too long".to_string()))?;
        let rc = unsafe { lode_simple_set_jieba_dict(path.as_ptr(), path_len) };
        if rc != SQLITE_OK {
            return Err(native_error(rc));
        }
    }

    Ok(())
}

/// Tokenize document text using the registered physical `simple` tokenizer.
///
/// # Errors
///
/// Returns an error when the input contains an embedded NUL, the native
/// callback reports invalid UTF-8 or offsets, or SQLite rejects tokenization.
pub fn tokenize_document(conn: &Connection, text: &str) -> Result<Vec<Token>, Error> {
    let input = CString::new(text).map_err(|error| Error::InvalidInput(error.to_string()))?;
    let text_len = i32::try_from(input.as_bytes().len())
        .map_err(|_| Error::InvalidInput("input is too large for the FTS5 API".to_string()))?;
    let mut collector = Collector {
        input: text,
        tokens: Vec::new(),
        error: None,
    };
    let rc = unsafe {
        lode_simple_tokenize(
            conn.handle(),
            input.as_ptr(),
            text_len,
            (&raw mut collector).cast(),
            collect_token,
        )
    };
    if let Some(error) = collector.error {
        return Err(Error::Callback(error));
    }
    if rc != SQLITE_OK {
        return Err(sqlite_error(unsafe { conn.handle() }, rc));
    }
    Ok(collector.tokens)
}

/// Return the directory containing the verified, materialized jieba files.
///
/// # Errors
///
/// Returns an error when the user cache location is unavailable, a cache file
/// cannot be written, or a concurrent writer leaves an invalid final cache.
pub fn ensure_jieba_dictionary() -> Result<PathBuf, Error> {
    ensure_jieba_dictionary_at(&cache_root()?.join("lode").join("native-tokenizer"))
}

fn ensure_jieba_dictionary_at(root: &Path) -> Result<PathBuf, Error> {
    let digest = dictionary_digest();
    let final_dir = root.join(format!("{DICTIONARY_VERSION}-{digest}"));
    if directory_is_valid(&final_dir, &digest)? {
        return Ok(final_dir);
    }

    fs::create_dir_all(root)?;
    let temporary_dir = create_temporary_directory(root)?;
    let write_result = (|| {
        for (name, contents) in DICTIONARY_FILES {
            let path = temporary_dir.join(name);
            fs::write(path, contents)?;
        }
        fs::write(temporary_dir.join(DICTIONARY_MARKER), digest.as_bytes())?;
        Ok::<(), io::Error>(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&temporary_dir);
        return Err(error.into());
    }

    match fs::rename(&temporary_dir, &final_dir) {
        Ok(()) => Ok(final_dir),
        Err(_error) if final_dir.exists() => {
            let _ = fs::remove_dir_all(&temporary_dir);
            if directory_is_valid(&final_dir, &digest)? {
                Ok(final_dir)
            } else {
                Err(Error::Cache(format!(
                    "dictionary cache exists but is incomplete: {}",
                    final_dir.display()
                )))
            }
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary_dir);
            Err(error.into())
        }
    }
}

fn cache_root() -> Result<PathBuf, Error> {
    if let Some(path) = std::env::var_os("LODE_CACHE_DIR") {
        if path.is_empty() {
            return Err(Error::Cache("LODE_CACHE_DIR is empty".to_string()));
        }
        return Ok(PathBuf::from(path));
    }
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| Error::Cache("LOCALAPPDATA is not set".to_string()))
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join("Library/Caches"))
            .ok_or_else(|| Error::Cache("HOME is not set".to_string()))
    } else if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".cache"))
            .ok_or_else(|| Error::Cache("HOME is not set".to_string()))
    }
}

fn dictionary_digest() -> String {
    let mut hasher = blake3::Hasher::new();
    for (name, contents) in DICTIONARY_FILES {
        hasher.update(name.as_bytes());
        hasher.update(contents);
    }
    hasher.finalize().to_hex().to_string()
}

fn directory_is_valid(directory: &Path, digest: &str) -> Result<bool, Error> {
    if !directory.is_dir() {
        return Ok(false);
    }
    let marker = match fs::read_to_string(directory.join(DICTIONARY_MARKER)) {
        Ok(marker) => marker,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if marker != digest {
        return Ok(false);
    }
    for (name, expected) in DICTIONARY_FILES {
        match fs::read(directory.join(name)) {
            Ok(actual) if actual == *expected => {}
            Ok(_) => return Ok(false),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(true)
}

fn create_temporary_directory(root: &Path) -> Result<PathBuf, Error> {
    for _ in 0..64 {
        let id = TEMP_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(".tmp-{}-{id}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(Error::Cache(
        "could not allocate a temporary dictionary directory".to_string(),
    ))
}

fn path_to_cstring(path: &Path) -> Result<CString, Error> {
    let text = path.to_str().ok_or_else(|| {
        Error::Cache(format!(
            "dictionary path is not valid UTF-8: {}",
            path.display()
        ))
    })?;
    let mut text = text.to_owned();
    if !text.ends_with(std::path::MAIN_SEPARATOR) {
        text.push(std::path::MAIN_SEPARATOR);
    }
    CString::new(text).map_err(|error| Error::Cache(error.to_string()))
}

fn sqlite_error(handle: *mut rusqlite::ffi::sqlite3, code: i32) -> Error {
    let message = unsafe {
        if handle.is_null() {
            "SQLite connection is null".to_string()
        } else {
            CStr::from_ptr(rusqlite::ffi::sqlite3_errmsg(handle))
                .to_string_lossy()
                .into_owned()
        }
    };
    Error::Sqlite { code, message }
}

fn native_error(code: i32) -> Error {
    let message = unsafe { CStr::from_ptr(lode_simple_last_error()) }
        .to_string_lossy()
        .into_owned();
    Error::Sqlite { code, message }
}

struct Collector<'a> {
    input: &'a str,
    tokens: Vec<Token>,
    error: Option<String>,
}

unsafe extern "C" fn collect_token(
    ctx: *mut libc::c_void,
    flags: libc::c_int,
    token: *const libc::c_char,
    token_len: libc::c_int,
    start: libc::c_int,
    end: libc::c_int,
) -> libc::c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let collector = unsafe { &mut *(ctx.cast::<Collector<'_>>()) };
        if token_len < 0 || start < 0 || end < start || flags < 0 {
            collector.error = Some("tokenizer returned invalid token bounds".to_string());
            return SQLITE_ERROR;
        }
        let Ok(token_len) = usize::try_from(token_len) else {
            collector.error = Some("tokenizer returned an invalid token length".to_string());
            return SQLITE_ERROR;
        };
        let Ok(start) = usize::try_from(start) else {
            collector.error = Some("tokenizer returned an invalid token start".to_string());
            return SQLITE_ERROR;
        };
        let Ok(end) = usize::try_from(end) else {
            collector.error = Some("tokenizer returned an invalid token end".to_string());
            return SQLITE_ERROR;
        };
        if token.is_null()
            || end > collector.input.len()
            || start > end
            || !collector.input.is_char_boundary(start)
            || !collector.input.is_char_boundary(end)
        {
            collector.error = Some("tokenizer returned invalid UTF-8 byte offsets".to_string());
            return SQLITE_ERROR;
        }
        let token_bytes = unsafe { std::slice::from_raw_parts(token.cast::<u8>(), token_len) };
        let Ok(text) = std::str::from_utf8(token_bytes) else {
            collector.error = Some("tokenizer returned non-UTF-8 token text".to_string());
            return SQLITE_ERROR;
        };
        let text = text.to_string();
        let flags = flags.cast_unsigned();
        if flags & TOKEN_COLOCATED != 0 && collector.tokens.is_empty() {
            collector.error = Some("tokenizer emitted a colocated token first".to_string());
            return SQLITE_ERROR;
        }
        collector.tokens.push(Token {
            flags,
            text,
            start,
            end,
        });
        SQLITE_OK
    }));
    match result {
        Ok(code) => code,
        Err(_) => SQLITE_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn dictionary_digest_is_stable_and_nonempty() {
        let digest = dictionary_digest();
        assert_eq!(digest.len(), 64);
        assert!(
            digest
                .chars()
                .all(|character| character.is_ascii_hexdigit())
        );
    }

    #[test]
    fn path_to_cstring_adds_a_separator() {
        let path = path_to_cstring(Path::new("/tmp/jieba")).unwrap();
        assert_eq!(
            path.to_str().unwrap(),
            format!("/tmp/jieba{}", std::path::MAIN_SEPARATOR)
        );
    }

    #[test]
    fn native_document_tokenizer_emits_colocated_pinyin() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let tokens = tokenize_document(&conn, "知识").unwrap();
        assert_eq!(tokens[0].text, "知");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, "知".len());
        assert_eq!(tokens[1].text, "z");
        assert_eq!(tokens[1].flags & TOKEN_COLOCATED, TOKEN_COLOCATED);
        assert_eq!(tokens[1].start, tokens[0].start);
        assert_eq!(tokens[1].end, tokens[0].end);
    }

    #[test]
    fn native_document_tokenizer_preserves_utf8_offsets_and_skips_whitespace() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let input = " A 你!🙂 ";
        let tokens = tokenize_document(&conn, input).unwrap();
        assert_eq!(tokens[0].text, "a");
        assert_eq!(tokens[0].start, 1);
        assert_eq!(tokens[0].end, 2);
        assert!(tokens.iter().any(|token| token.text == "你"));
        assert!(tokens.iter().any(|token| token.text == "!"));
        assert!(tokens.iter().any(|token| token.text == "🙂"));
        assert!(
            tokens
                .iter()
                .all(|token| !token.text.chars().all(char::is_whitespace))
        );
    }

    #[test]
    fn callback_rejects_a_colocated_first_token() {
        let mut collector = Collector {
            input: "知",
            tokens: Vec::new(),
            error: None,
        };
        let token = CString::new("zhi").unwrap();
        let rc = unsafe {
            collect_token(
                (&raw mut collector).cast(),
                i32::try_from(TOKEN_COLOCATED).unwrap(),
                token.as_ptr(),
                3,
                0,
                3,
            )
        };
        assert_eq!(rc, SQLITE_ERROR);
        assert!(collector.error.is_some());
    }

    #[test]
    fn embedded_nul_is_rejected_before_ffi() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let error = tokenize_document(&conn, "a\0b").unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)));
    }

    #[test]
    fn native_tokenizer_can_create_an_fts_table() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        conn.execute(
            "CREATE VIRTUAL TABLE docs USING fts5(text, tokenize='simple')",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO docs(text) VALUES (?1)", ["知识"])
            .unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM docs WHERE docs MATCH simple_query(?1)",
                ["zhi"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        conn.execute(
            "CREATE VIRTUAL TABLE docs_vocab USING fts5vocab(docs, instance)",
            [],
        )
        .unwrap();
        let positions = conn
            .prepare("SELECT term, offset FROM docs_vocab WHERE doc = 1 ORDER BY offset, term")
            .unwrap()
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            positions,
            vec![
                ("z".to_string(), 0),
                ("zhi".to_string(), 0),
                ("知".to_string(), 0),
                ("s".to_string(), 1),
                ("shi".to_string(), 1),
                ("z".to_string(), 1),
                ("zhi".to_string(), 1),
                ("识".to_string(), 1),
            ]
        );
    }

    #[test]
    fn native_query_helper_ignores_leading_whitespace() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        conn.execute(
            "CREATE VIRTUAL TABLE docs USING fts5(text, tokenize='simple')",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO docs(text) VALUES (?1)", ["知识"])
            .unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM docs WHERE docs MATCH simple_query(?1)",
                [" \t知识"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn native_query_helper_handles_null_flags_without_crashing() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let result: String = conn
            .query_row("SELECT simple_query('hello', NULL)", [], |row| row.get(0))
            .unwrap();
        let default_result: String = conn
            .query_row("SELECT simple_query('hello')", [], |row| row.get(0))
            .unwrap();
        assert_eq!(result, default_result);
    }

    #[test]
    fn native_query_helper_rejects_truncated_utf8() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let error = conn
            .query_row::<String, _, _>("SELECT simple_query(CAST(x'F0' AS TEXT))", [], |row| {
                row.get(0)
            })
            .unwrap_err();
        assert!(error.to_string().contains("truncated UTF-8 input"));
    }

    #[test]
    fn jieba_query_requires_controlled_dictionary_initialization() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let error = conn
            .query_row::<String, _, _>("SELECT jieba_query('知识')", [], |row| row.get(0))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("jieba dictionary has not been configured")
        );
    }

    #[test]
    fn mutable_dictionary_helpers_are_not_registered() {
        let conn = Connection::open_in_memory().unwrap();
        register_connection(&conn, false).unwrap();
        let error = conn
            .execute("SELECT pinyin_dict('custom.txt')", [])
            .unwrap_err();
        assert!(error.to_string().contains("no such function"));
        let error = conn
            .execute("SELECT jieba_dict('custom/')", [])
            .unwrap_err();
        assert!(error.to_string().contains("no such function"));
    }

    #[test]
    fn invalid_jieba_dictionary_is_returned_as_an_error() {
        let root = tempfile::tempdir().unwrap();
        let path = path_to_cstring(root.path()).unwrap();
        let path_len = i32::try_from(path.as_bytes().len()).unwrap();
        let rc = unsafe { lode_simple_set_jieba_dict(path.as_ptr(), path_len) };
        assert_eq!(rc, SQLITE_ERROR);
        let message = unsafe { CStr::from_ptr(lode_simple_last_error()) }.to_string_lossy();
        assert!(message.contains("open") || message.contains("failed"));
    }

    #[test]
    fn dictionary_cache_is_atomic_across_concurrent_initializers() {
        let root = tempfile::tempdir().unwrap();
        let paths = std::thread::scope(|scope| {
            (0..4)
                .map(|_| scope.spawn(|| ensure_jieba_dictionary_at(root.path()).unwrap()))
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(paths.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(directory_is_valid(&paths[0], &dictionary_digest()).unwrap());
    }

    #[test]
    fn corrupted_dictionary_cache_is_reported() {
        let root = tempfile::tempdir().unwrap();
        let path = ensure_jieba_dictionary_at(root.path()).unwrap();
        fs::write(path.join("user.dict.utf8"), b"corrupted").unwrap();
        let error = ensure_jieba_dictionary_at(root.path()).unwrap_err();
        assert!(error.to_string().contains("incomplete"));
    }

    #[test]
    fn unavailable_dictionary_cache_root_is_reported() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("cache-root");
        fs::write(&file, b"not a directory").unwrap();
        let error = ensure_jieba_dictionary_at(&file).unwrap_err();
        assert!(matches!(error, Error::Io(_)));
    }
}
