# Vendored native tokenizer sources

This directory contains the fixed native-tokenizer inputs used by the Rust
build. The build script never downloads source or dictionaries.

- `simple-0.7.1/` is the upstream `wangfenjin/simple` v0.7.1 source. The
  `pinyin.cc` resource loader and FTS5 error handling are adapted locally;
  `contrib/pinyin.txt` is embedded into the build output.
- `cppjieba/` is the exact commit pinned by simple v0.7.1:
  `194c144d8b5ed1baf3190d07c5226e804454ab47`. Its fatal checks are adapted
  to throw C++ exceptions across the Rust boundary, and dictionary file
  opening accepts UTF-8 paths on Windows.
- `dict/` contains the matching cppjieba dictionary files. They are embedded
  in the Rust crate and materialized into a versioned user cache only when the
  `jieba` query strategy is selected.
- `sqlite/` contains SQLite public headers copied from the `libsqlite3-sys`
  bundled build used by rusqlite. SQLite implementation code is not copied;
  the native tokenizer links against rusqlite's single bundled SQLite.
- The Lode build intentionally does not register upstream's mutable
  `pinyin_dict()` or `jieba_dict()` SQL functions. Dictionary initialization is
  controlled by the Rust wrapper and uses only the embedded, verified files.
  Calling `jieba_query()` before that initialization returns an error instead
  of falling back to a working-directory `./dict/` directory.

The upstream licenses are retained alongside their source trees. The Rust
wrapper and local adaptations are licensed under the repository license.
