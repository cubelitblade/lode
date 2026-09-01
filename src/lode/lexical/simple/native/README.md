# Native `simple` extension

The `simple` FTS5 tokenizer (https://github.com/wangfenjin/simple) is a C
extension that indexes each Han character plus its pinyin, and provides
`simple_query()` / `jieba_query()` helpers for query construction.

## Layout

The shared library is platform-specific and ships in the `lode-simple-native`
distribution, selected by pip via wheel tags. This package only holds the jieba
dictionary:

- `dict/` — the jieba dictionary (identical across platforms) used by
  `jieba_query()`.

The platform binary is imported from `lode_simple_native` (see `__init__.py`).

## Origin

All binaries and `dict/` were extracted from the v0.7.1 release assets
(https://github.com/wangfenjin/simple/releases/tag/v0.7.1):

- `libsimple-linux-ubuntu-22.04.zip`
- `libsimple-osx-arm64.zip`, `libsimple-osx-x64.zip`
- `libsimple-windows-arm64.zip`, `libsimple-windows-x64.zip`

## Notes

- `load_simple` (in `__init__.py`) loads the platform binary from
  `lode_simple_native` and points jieba at `dict/`.
- The `dict/` files are binary artifacts; they are not meant to be edited.
