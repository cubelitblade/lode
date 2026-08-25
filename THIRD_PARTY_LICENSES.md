# Third party licenses

This document lists third-party components included in Lode's distribution, including vendored source code and bundled binaries.

## Overview

| Author         | Name                     | Version | Distribution type | License |
| -------------- | ------------------------ | ------- | ----------------- | ------- |
| LangChain, Inc | langchain_text_splitters | 1.1.2   | vendored          | MIT     |
| Wang Fenjin    | simple                   | 0.7.1   | bundled           | MIT     |

## Details

### langchain_text_splitters

- **Description**: LangChain Text Splitters contains utilities for splitting into chunks a wide variety of text documents.
- **Source**: https://github.com/langchain-ai/langchain
- **Path**: `libs/text-splitters`
- **Revision**: `a2e53fda733c3ccc180b85fbbdb48893d770549f`
- **Version**: 1.1.2
- **Copyright**: LangChain, Inc.
- **License**: MIT

#### How it is used

This project vendors `langchain_text_splitters` under `src/lode/_vendor/langchain_text_splitters`.

The code is included to provide text splitting functionality without requiring the LangChain runtime dependency stack.

#### License

The original license is available at https://github.com/langchain-ai/langchain/blob/master/LICENSE.

A copy of the license text is included below.

```
MIT License

Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### simple

- **Description**: A SQLite3 fts5 tokenizer which supports Chinese and PinYin.
- **Source**: https://github.com/wangfenjin/simple
- **Revision**: `4ed008934495fc55ff4bf6620bba58311988b23e`
- **Version**: 0.7.1
- **Copyright**: Wang Fenjin
- **License**: MIT

#### How it is used

This project bundles prebuilt binaries under `src/lode/lexical/simple/native`.

The binary is included to provide Chinese text tokenization support.

#### License

The original license is available at https://github.com/wangfenjin/simple/blob/master/LICENSE.

The original project is dual-licensed under MIT and GPL-3.0-or-later. This project uses the MIT license option.

A copy of the MIT portion of the license text is included below.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
