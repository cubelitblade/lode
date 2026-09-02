//! lode FTS5 simple tokenizer (cdylib, pinyin folding).
//!
//! Exports `#[no_mangle] extern "C"` FTS5 tokenizer functions. The FTS5 C API
//! surface is filled in during lexical integration.

/// Placeholder symbol so the cdylib links until the real C API lands.
#[unsafe(no_mangle)]
pub extern "C" fn lode_simple_native_version() -> u32 {
    0
}
