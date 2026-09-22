#include "simple_tokenizer.h"

#include <cstring>
#include <exception>
#include <string>

extern "C" int sqlite3_simple_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

namespace {
thread_local std::string last_error;

int set_error(int code, const char *message) {
  last_error = message;
  return code;
}

int set_error(int code, const std::exception &error) {
  return set_error(code, error.what());
}
}  // namespace

extern "C" const char *lode_simple_last_error() { return last_error.c_str(); }

static int fts5_api_from_db(sqlite3 *db, fts5_api **api) {
  sqlite3_stmt *stmt = nullptr;
  *api = nullptr;
  int rc = sqlite3_prepare(db, "SELECT fts5(?1)", -1, &stmt, nullptr);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_pointer(stmt, 1, api, "fts5_api_ptr", nullptr);
    if (rc == SQLITE_OK) (void)sqlite3_step(stmt);
    int finalize_rc = sqlite3_finalize(stmt);
    if (rc == SQLITE_OK) rc = finalize_rc;
  }
  return rc;
}

extern "C" int lode_simple_register(sqlite3 *db) {
  last_error.clear();
  if (db == nullptr) return SQLITE_MISUSE;
  try {
    return sqlite3_simple_init(db, nullptr, nullptr);
  } catch (const std::exception &error) {
    return set_error(SQLITE_ERROR, error);
  } catch (...) {
    return set_error(SQLITE_ERROR, "native tokenizer registration failed");
  }
}

extern "C" int lode_simple_set_jieba_dict(const char *path, int path_len) {
  last_error.clear();
  if (path == nullptr || path_len <= 0) return set_error(SQLITE_MISUSE, "invalid jieba dictionary path");
  try {
    if (!simple_tokenizer::set_jieba_dict_path(std::string(path, static_cast<size_t>(path_len)))) {
      return set_error(SQLITE_ERROR, "jieba dictionary is already initialized with another path");
    }
    return SQLITE_OK;
  } catch (const std::exception &error) {
    return set_error(SQLITE_ERROR, error);
  } catch (...) {
    return set_error(SQLITE_ERROR, "jieba dictionary initialization failed");
  }
}

extern "C" int lode_simple_tokenize(sqlite3 *db, const char *text, int text_len, void *ctx,
                                     int (*callback)(void *, int, const char *, int, int, int)) {
  last_error.clear();
  if (db == nullptr || text == nullptr || text_len < 0 || callback == nullptr) return SQLITE_MISUSE;
  try {
    fts5_api *fts5api = nullptr;
    int rc = fts5_api_from_db(db, &fts5api);
    if (rc != SQLITE_OK || fts5api == nullptr || fts5api->iVersion < 2) return SQLITE_ERROR;

    void *user_data = nullptr;
    fts5_tokenizer tokenizer = {};
    rc = fts5api->xFindTokenizer(fts5api, "simple", &user_data, &tokenizer);
    if (rc != SQLITE_OK) return rc;
    Fts5Tokenizer *instance = nullptr;
    rc = tokenizer.xCreate(user_data, nullptr, 0, &instance);
    if (rc != SQLITE_OK) return rc;
    rc = tokenizer.xTokenize(instance, ctx, FTS5_TOKENIZE_DOCUMENT, text, text_len, callback);
    tokenizer.xDelete(instance);
    return rc;
  } catch (const std::exception &error) {
    return set_error(SQLITE_ERROR, error);
  } catch (...) {
    return set_error(SQLITE_ERROR, "native tokenizer callback failed");
  }
}
