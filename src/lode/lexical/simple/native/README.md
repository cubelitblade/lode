# Native `simple` extension

The `simple` FTS5 tokenizer (https://github.com/wangfenjin/simple) is a C
extension that indexes each Han character plus its pinyin, and provides
`simple_query()` / `jieba_query()` helpers for query construction.

## Layout

The shared library is platform-specific; each OS/architecture has its own
binary under a matching subdirectory:

- `linux/libsimple.so`
- `darwin/arm64/libsimple.dylib`, `darwin/x86_64/libsimple.dylib`
- `win32/arm64/simple.dll`, `win32/x86_64/simple.dll`

`dict/` is the jieba dictionary (identical across platforms) used by
`jieba_query()`.

## Origin

All binaries and `dict/` were extracted from the v0.7.1 release assets
(https://github.com/wangfenjin/simple/releases/tag/v0.7.1):

- `libsimple-linux-ubuntu-22.04.zip`
- `libsimple-osx-arm64.zip`, `libsimple-osx-x64.zip`
- `libsimple-windows-arm64.zip`, `libsimple-windows-x64.zip`

## Notes

- `load_simple` (in `__init__.py`) picks the binary for the current platform
  and raises a clear error when none is bundled.
- The binaries and `dict/` are binary artifacts; they are not meant to be
  edited.