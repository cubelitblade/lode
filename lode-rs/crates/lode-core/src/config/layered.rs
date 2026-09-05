//! Layered config loading: combine defaults, TOML, env and kwargs.
//!
//! Sources ascend in precedence: compiled defaults < user config file <
//! project config files (< env vars, < constructor kwargs — the latter two
//! arrive in later phases). Within the file family, later-discovered files
//! win, so project-local files override the user file, matching Python's
//! ordering.
//!
//! Discovery follows `docs/configuration.md`: a user-level file under the
//! platform config dir, then the project files `.lode.toml`, `lode.toml`,
//! and `.lode/config.toml`. Existing files are deep-merged in ascending
//! precedence and decoded into [`Settings`]; missing files are not an error.

use std::path::{Path, PathBuf};

use crate::config::Settings;

/// Relative project config filenames, lowest precedence first.
const PROJECT_CONFIG_PATHS: [&str; 3] = [".lode.toml", "lode.toml", ".lode/config.toml"];

/// Directory holding the user-level config file.
fn user_config_dir() -> PathBuf {
    // Mirror Python's `platformdirs.user_config_dir("lode")` reasonably:
    // honour `XDG_CONFIG_HOME`, falling back to `~/.config`.
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(dir).join("lode");
    }
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".config").join("lode"))
        .unwrap_or_else(|| PathBuf::from(".lode"))
}

/// Candidate config files in ascending precedence (user, then project).
fn discover_config_files(base: &Path, user_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![user_dir.join("config.toml")];
    for name in PROJECT_CONFIG_PATHS {
        candidates.push(base.join(name));
    }
    candidates.into_iter().filter(|p| p.is_file()).collect()
}

/// Merge a dotted-key-free TOML table into `acc`, overriding scalars and
/// recursing into equal-named tables.
fn merge(a: &mut toml::value::Table, b: &toml::value::Table) {
    for (key, val) in b {
        match (a.get_mut(key), val) {
            (Some(toml::Value::Table(at)), toml::Value::Table(bt)) => merge(at, bt),
            _ => {
                a.insert(key.clone(), val.clone());
            }
        }
    }
}

/// Parse a TOML file into a table, treating malformed files as an error.
fn parse_file(path: &Path) -> Result<toml::value::Table, crate::Error> {
    let text = std::fs::read_to_string(path)?;
    text.parse::<toml::Value>()
        .map_err(|e| crate::Error::Config(format!("invalid TOML in {}: {e}", path.display())))?
        .as_table()
        .cloned()
        .ok_or_else(|| {
            crate::Error::Config(format!(
                "expected a TOML table at top level in {}",
                path.display()
            ))
        })
}

/// Load settings from the given config roots.
///
/// Merges the user config (`user_dir/config.toml`) and the project files
/// under `base` in ascending precedence; anything unspecified falls back to
/// [`Settings::default`]. Malformed files propagate as
/// [`crate::Error::Config`].
fn load_settings_from(base: &Path, user_dir: &Path) -> Result<Settings, crate::Error> {
    let mut acc = toml::value::Table::new();
    for path in discover_config_files(base, user_dir) {
        let tbl = parse_file(&path)?;
        merge(&mut acc, &tbl);
    }
    let settings: Settings = acc
        .try_into()
        .map_err(|e| crate::Error::Config(format!("cannot interpret config: {e}")))?;
    Ok(settings)
}

/// Load settings following the configured precedence chain.
///
/// Discovers and merges the user + project config files rooted at `base`;
/// anything unspecified falls back to [`Settings::default`]. Malformed
/// files propagate as [`crate::Error::Config`].
pub fn load_settings_for(base: &Path) -> Result<Settings, crate::Error> {
    load_settings_from(base, &user_config_dir())
}

/// Convenience overload rooting discovery at the current directory.
pub fn load_settings() -> Result<Settings, crate::Error> {
    load_settings_for(Path::new("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EmbeddingProvider;

    #[test]
    fn defaults_when_no_config_exists() {
        let dir = tempfile::tempdir().unwrap();
        let settings = load_settings_from(dir.path(), dir.path()).unwrap();
        assert_eq!(settings.embedding.model_dimension, None);
        assert_eq!(settings.chunking.size, 1024);
        assert_eq!(settings.chunking.overlap, 128);
        assert_eq!(settings.fts.strategy, "simple");
    }

    #[test]
    fn reads_model_dimension_from_project_toml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lode.toml"),
            "[embedding]\nmodel_dimension = 768\n",
        )
        .unwrap();
        let settings = load_settings_from(dir.path(), dir.path()).unwrap();
        assert_eq!(settings.embedding.model_dimension, Some(768));
    }

    #[test]
    fn lower_precedence_user_is_overridden_by_project() {
        let dir = tempfile::tempdir().unwrap();
        let user_dir = dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();
        std::fs::write(
            user_dir.join("config.toml"),
            "[embedding]\nmodel_dimension = 384\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("lode.toml"),
            "[embedding]\nmodel_dimension = 640\n",
        )
        .unwrap();

        let settings = load_settings_from(dir.path(), &user_dir).unwrap();
        assert_eq!(settings.embedding.model_dimension, Some(640));
    }

    #[test]
    fn later_project_file_wins() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".lode.toml"), "[chunking]\nsize = 512\n").unwrap();
        std::fs::write(dir.path().join("lode.toml"), "[chunking]\nsize = 2048\n").unwrap();

        let settings = load_settings_from(dir.path(), dir.path()).unwrap();
        assert_eq!(settings.chunking.size, 2048);
    }

    #[test]
    fn rejects_malformed_toml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lode.toml"), "[[[\n").unwrap();
        let err = load_settings_from(dir.path(), dir.path()).unwrap_err();
        assert!(err.to_string().contains("invalid TOML"));
    }

    #[test]
    fn embedding_defaults_match_python() {
        let dir = tempfile::tempdir().unwrap();
        let settings = load_settings_from(dir.path(), dir.path()).unwrap();
        let e = &settings.embedding;
        assert_eq!(e.provider, EmbeddingProvider::OpenAICompatible);
        assert_eq!(e.model, None);
        assert_eq!(e.model_dimension, None);
        assert_eq!(e.output_dimension, None);
        assert_eq!(e.batch_size, 32);
        assert_eq!(
            e.openai_compatible.http.endpoint,
            "https://api.openai.com/v1"
        );
        assert_eq!(e.tei_native.http.endpoint, "http://localhost:8080");
        assert!(!e.tei_native.truncate);
        assert_eq!(e.tei_native.truncation_direction, "right");
        assert_eq!(e.ollama.http.endpoint, "http://localhost:11434");
        assert!(e.ollama.truncate);
    }

    #[test]
    fn reads_full_embedding_config_from_toml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lode.toml"),
            r#"
[embedding]
provider = "ollama"
model = "nomic-embed-text"
model_dimension = 768
output_dimension = 512
batch_size = 16

[embedding.ollama]
endpoint = "http://localhost:11435"
key = "secret"
max_retries = 5
timeout = 30.0
truncate = false
"#,
        )
        .unwrap();
        let settings = load_settings_from(dir.path(), dir.path()).unwrap();
        let e = &settings.embedding;
        assert_eq!(e.provider, EmbeddingProvider::Ollama);
        assert_eq!(e.model.as_deref(), Some("nomic-embed-text"));
        assert_eq!(e.model_dimension, Some(768));
        assert_eq!(e.output_dimension, Some(512));
        assert_eq!(e.batch_size, 16);
        assert_eq!(e.ollama.http.endpoint, "http://localhost:11435");
        assert_eq!(e.ollama.http.key.as_deref(), Some("secret"));
        assert_eq!(e.ollama.http.max_retries, 5);
        assert_eq!(e.ollama.http.timeout, 30.0);
        assert!(!e.ollama.truncate);
    }
}
