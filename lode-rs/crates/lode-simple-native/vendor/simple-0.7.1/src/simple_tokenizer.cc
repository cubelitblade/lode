#include "simple_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

namespace simple_tokenizer {
namespace {
std::mutex pinyin_mutex;
std::shared_ptr<PinYin> global_pinyin;
#ifdef USE_JIEBA
std::mutex jieba_mutex;
std::unique_ptr<cppjieba::Jieba> global_jieba;
std::string configured_jieba_dict_path;
#endif
}

SimpleTokenizer::SimpleTokenizer(const char **azArg, int nArg) {
  if (nArg >= 1) {
    enable_pinyin = atoi(azArg[0]) != 0;
  }
}

std::shared_ptr<PinYin> SimpleTokenizer::get_pinyin() {
  std::lock_guard<std::mutex> lock(pinyin_mutex);
  if (global_pinyin == nullptr) {
    global_pinyin = std::make_shared<PinYin>();
  }
  return global_pinyin;
}

bool SimpleTokenizer::set_pinyin_dict(const std::string &pinyin_file_path, std::string &err) {
  std::shared_ptr<PinYin> new_pinyin;
  try {
    if (pinyin_file_path.empty()) {
      new_pinyin = std::make_shared<PinYin>();
    } else {
      new_pinyin = std::make_shared<PinYin>(pinyin_file_path);
    }
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }

  std::lock_guard<std::mutex> lock(pinyin_mutex);
  global_pinyin = new_pinyin;
  return true;
}

static TokenCategory from_char(char c) {
  auto uc = static_cast<unsigned char>(c);
  // ASCII should in 0..127
  if (uc > 127) {
    return TokenCategory::OTHER;
  }
  if (std::isdigit(uc)) {
    return TokenCategory::DIGIT;
  }
  if (std::isspace(uc) || std::iscntrl(uc)) {
    return TokenCategory::SPACE;
  }
  if (std::isalpha(uc)) {
    return TokenCategory::ASCII_ALPHABETIC;
  }
  return TokenCategory::OTHER;
}

static int utf8_char_length(const char *text, int textLen, int index) {
  const auto first = static_cast<unsigned char>(text[index]);
  int length = 0;
  if (first <= 0x7f) {
    return 1;
  } else if (first >= 0xc2 && first <= 0xdf) {
    length = 2;
  } else if (first >= 0xe0 && first <= 0xef) {
    length = 3;
  } else if (first >= 0xf0 && first <= 0xf4) {
    length = 4;
  } else {
    throw std::runtime_error("invalid UTF-8 input");
  }

  if (length > textLen - index) {
    throw std::runtime_error("truncated UTF-8 input");
  }
  for (int i = 1; i < length; ++i) {
    const auto continuation = static_cast<unsigned char>(text[index + i]);
    if ((continuation & 0xc0) != 0x80) {
      throw std::runtime_error("invalid UTF-8 input");
    }
  }

  std::uint32_t codepoint = first & (length == 2 ? 0x1f : length == 3 ? 0x0f : 0x07);
  for (int i = 1; i < length; ++i) {
    codepoint = (codepoint << 6) | (static_cast<unsigned char>(text[index + i]) & 0x3f);
  }
  const std::uint32_t minimum = length == 2 ? 0x80 : length == 3 ? 0x800 : 0x10000;
  if (codepoint < minimum || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
    throw std::runtime_error("invalid UTF-8 input");
  }
  return length;
}

static void validate_utf8(const char *text, int textLen) {
  for (int index = 0; index < textLen;) {
    index += utf8_char_length(text, textLen, index);
  }
}

std::string SimpleTokenizer::tokenize_query(const char *text, int textLen, int flags) {
  int start = 0;
  int index = 0;
  std::string tmp;
  std::string result;
  while (index < textLen) {
    TokenCategory category = from_char(text[index]);
    switch (category) {
      case TokenCategory::OTHER:
        index += utf8_char_length(text, textLen, index);
        break;
      default:
        while (++index < textLen && from_char(text[index]) == category) {
        }
        break;
    }
    tmp.clear();
    std::copy(text + start, text + index, std::back_inserter(tmp));
    append_result(result, tmp, category, flags);
    start = index;
  }
  return result;
}

#ifdef USE_JIEBA
std::string jieba_dict_path;

static void initialize_jieba_locked() {
  if (global_jieba == nullptr) {
    if (jieba_dict_path.empty()) {
      throw std::runtime_error("jieba dictionary has not been configured");
    }
    global_jieba = std::make_unique<cppjieba::Jieba>(jieba_dict_path + "jieba.dict.utf8",
                                                     jieba_dict_path + "hmm_model.utf8",
                                                     jieba_dict_path + "user.dict.utf8",
                                                     jieba_dict_path + "idf.utf8",
                                                     jieba_dict_path + "stop_words.utf8");
    configured_jieba_dict_path = jieba_dict_path;
  }
}

bool set_jieba_dict_path(const std::string &path) {
  if (path.empty()) return false;
  std::lock_guard<std::mutex> lock(jieba_mutex);
  if (global_jieba != nullptr && configured_jieba_dict_path != path) return false;
  if (global_jieba == nullptr) {
    auto new_jieba = std::make_unique<cppjieba::Jieba>(path + "jieba.dict.utf8", path + "hmm_model.utf8",
                                                       path + "user.dict.utf8", path + "idf.utf8",
                                                       path + "stop_words.utf8");
    jieba_dict_path = path;
    configured_jieba_dict_path = path;
    global_jieba = std::move(new_jieba);
  }
  return true;
}

static cppjieba::Jieba *get_jieba() {
  std::lock_guard<std::mutex> lock(jieba_mutex);
  initialize_jieba_locked();
  return global_jieba.get();
}

std::string SimpleTokenizer::tokenize_jieba_query(const char *text, int textLen, int flags) {
  validate_utf8(text, textLen);
  std::string input(text, static_cast<size_t>(textLen));
  std::string tmp;
  std::string result;
  std::vector<cppjieba::Word> words;
  get_jieba()->Cut(input, words);
  for (auto word : words) {
    if (word.offset >= input.size()) {
      throw std::runtime_error("jieba returned an invalid token offset");
    }
    // if all char is the same category, then use that category
    // otherwise use OTHER
    // fix https://github.com/wangfenjin/simple/issues/176
    TokenCategory category = from_char(text[word.offset]);
    for (auto c : word.word) {
      if (from_char(c) != category) {
        category = TokenCategory::OTHER;
        break;
      }
    }
    append_result(result, word.word, category, flags);
  }
  return result;
}
#endif

void SimpleTokenizer::append_result(std::string &result, std::string part, TokenCategory category, int flags) {
  if (category != TokenCategory::SPACE) {
    std::string tmp = std::move(part);
    if (category == TokenCategory::ASCII_ALPHABETIC) {
      std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    if (flags != 0 && category == TokenCategory::ASCII_ALPHABETIC && tmp.size() > 1) {
      if (result.empty()) {
        result.append("( ");
      } else {
        result.append(" AND ( ");
      }
      std::set<std::string> pys = SimpleTokenizer::get_pinyin()->split_pinyin(tmp);
      bool addOr = false;
      for (const std::string &s : pys) {
        if (addOr) {
          result.append(" OR ");
        }
        result.append(s);
        result.append("*");
        addOr = true;
      }
      result.append(" )");
    } else {
      if (!result.empty()) {
        result.append(" AND ");
      }
      if (tmp == "\"") {
        tmp += tmp;
      }
      if (category != TokenCategory::ASCII_ALPHABETIC) {
        result.append('"' + tmp + '"');
      } else {
        result.append(tmp);
      }
      if (category != TokenCategory::OTHER) {
        result.append("*");
      }
    }
  }
}

// https://cloud.tencent.com/developer/article/1198371
int SimpleTokenizer::tokenize(void *pCtx, int flags, const char *text, int textLen, xTokenFn xToken) const {
  int rc = SQLITE_OK;
  int start = 0;
  int index = 0;
  std::string result;
  while (index < textLen) {
    TokenCategory category = from_char(text[index]);
    switch (category) {
      case TokenCategory::OTHER:
        index += utf8_char_length(text, textLen, index);
        break;
      default:
        while (++index < textLen && from_char(text[index]) == category) {
        }
        break;
    }
    if (category != TokenCategory::SPACE) {
      result.clear();
      std::copy(text + start, text + index, std::back_inserter(result));
      if (category == TokenCategory::ASCII_ALPHABETIC) {
        std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
      }

      rc = xToken(pCtx, 0, result.c_str(), (int)result.length(), start, index);
      if (rc != SQLITE_OK) return rc;
      if (enable_pinyin && category == TokenCategory::OTHER && (flags & FTS5_TOKENIZE_DOCUMENT)) {
        std::shared_ptr<PinYin> pinyin = SimpleTokenizer::get_pinyin();
        const std::vector<std::string> &pys = pinyin->get_pinyin(result);
        for (const std::string &s : pys) {
          rc = xToken(pCtx, FTS5_TOKEN_COLOCATED, s.c_str(), (int)s.length(), start, index);
          if (rc != SQLITE_OK) return rc;
        }
      }
    }
    start = index;
  }
  return rc;
}
}  // namespace simple_tokenizer
