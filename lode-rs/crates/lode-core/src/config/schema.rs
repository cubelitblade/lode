//! The public configuration key schema shared by TOML, environment, and CLI
//! adapters.

use super::{EmbeddingHttpConfig, Settings};

/// The scalar representation accepted by a configuration key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    /// A boolean, including the human-friendly aliases accepted by lode.
    Bool,
    /// A required string.
    String,
    /// An optional string. `unset` removes it from the selected TOML layer.
    OptionalString,
    /// A non-negative 32-bit integer.
    U32,
    /// A non-negative platform-sized integer.
    Usize,
    /// A non-negative 64-bit integer.
    U64,
    /// A finite floating-point number.
    Float,
    /// A list of strings, encoded as JSON or comma-separated text.
    StringList,
    /// A string selected from a fixed set of values.
    Enum(&'static [&'static str]),
}

/// Description of one supported configuration leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigKey {
    /// Canonical dotted TOML path.
    pub path: &'static str,
    /// Accepted input representation.
    pub kind: ValueKind,
    /// Whether `config set` and `config unset` may edit it.
    pub editable: bool,
    /// Whether output must replace a non-null value with `[redacted]`.
    pub sensitive: bool,
}

const PALETTES: &[&str] = &["ansi", "accessible_tol", "accessible_okabe_ito"];
const PROVIDERS: &[&str] = &["openai_compatible", "tei_native", "ollama"];
const TRUNCATION_DIRECTIONS: &[&str] = &["left", "right"];
const NORM_KINDS: &[&str] = &["minmax", "softmax"];
const FUSION_KINDS: &[&str] = &["linear", "rrf"];
const FTS_STRATEGIES: &[&str] = &["unicode61", "trigram", "simple", "jieba"];

/// All supported configuration keys in stable display order.
pub const CONFIG_KEYS: &[ConfigKey] = &[
    ConfigKey {
        path: "version",
        kind: ValueKind::U32,
        editable: false,
        sensitive: false,
    },
    ConfigKey {
        path: "app.output.palette",
        kind: ValueKind::Enum(PALETTES),
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "app.output.no_color",
        kind: ValueKind::Bool,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "app.ignore.sources",
        kind: ValueKind::StringList,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.provider",
        kind: ValueKind::Enum(PROVIDERS),
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.model",
        kind: ValueKind::OptionalString,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.model_dimension",
        kind: ValueKind::U32,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.output_dimension",
        kind: ValueKind::U32,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.batch_size",
        kind: ValueKind::Usize,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.openai_compatible.endpoint",
        kind: ValueKind::String,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.openai_compatible.key",
        kind: ValueKind::OptionalString,
        editable: true,
        sensitive: true,
    },
    ConfigKey {
        path: "embedding.openai_compatible.max_retries",
        kind: ValueKind::U32,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.openai_compatible.timeout",
        kind: ValueKind::Float,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.tei_native.endpoint",
        kind: ValueKind::String,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.tei_native.key",
        kind: ValueKind::OptionalString,
        editable: true,
        sensitive: true,
    },
    ConfigKey {
        path: "embedding.tei_native.max_retries",
        kind: ValueKind::U32,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.tei_native.timeout",
        kind: ValueKind::Float,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.tei_native.truncate",
        kind: ValueKind::Bool,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.tei_native.truncation_direction",
        kind: ValueKind::Enum(TRUNCATION_DIRECTIONS),
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.ollama.endpoint",
        kind: ValueKind::String,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.ollama.key",
        kind: ValueKind::OptionalString,
        editable: true,
        sensitive: true,
    },
    ConfigKey {
        path: "embedding.ollama.max_retries",
        kind: ValueKind::U32,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.ollama.timeout",
        kind: ValueKind::Float,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "embedding.ollama.truncate",
        kind: ValueKind::Bool,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.top_k",
        kind: ValueKind::U32,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.norm.type",
        kind: ValueKind::Enum(NORM_KINDS),
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.norm.softmax.temperature",
        kind: ValueKind::Float,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.fusion.type",
        kind: ValueKind::Enum(FUSION_KINDS),
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.fusion.linear.semantic_factor",
        kind: ValueKind::Float,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.fusion.linear.lexical_factor",
        kind: ValueKind::Float,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "retrieval.fusion.rrf.k",
        kind: ValueKind::U64,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "chunking.size",
        kind: ValueKind::Usize,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "chunking.overlap",
        kind: ValueKind::Usize,
        editable: true,
        sensitive: false,
    },
    ConfigKey {
        path: "fts.strategy",
        kind: ValueKind::Enum(FTS_STRATEGIES),
        editable: true,
        sensitive: false,
    },
];

/// Find a key by its canonical dotted path.
#[must_use]
pub fn find_key(path: &str) -> Option<&'static ConfigKey> {
    CONFIG_KEYS.iter().find(|key| key.path == path)
}

/// Parse a CLI or environment string according to a known key's type.
pub fn parse_value(path: &str, raw: &str) -> crate::Result<toml::Value> {
    let key = find_key(path).ok_or_else(|| unknown_key(path))?;
    let value = match key.kind {
        ValueKind::Bool => toml::Value::Boolean(parse_bool(path, raw)?),
        ValueKind::String | ValueKind::OptionalString => toml::Value::String(raw.to_owned()),
        ValueKind::U32 => toml::Value::Integer(parse_integer::<u32>(path, raw)? as i64),
        ValueKind::Usize => {
            let value = parse_integer::<usize>(path, raw)?;
            toml::Value::Integer(i64::try_from(value).map_err(|_| integer_error(path))?)
        }
        ValueKind::U64 => {
            let value = parse_integer::<u64>(path, raw)?;
            toml::Value::Integer(i64::try_from(value).map_err(|_| integer_error(path))?)
        }
        ValueKind::Float => toml::Value::Float(parse_float(path, raw)?),
        ValueKind::StringList => toml::Value::Array(parse_string_list(path, raw)?),
        ValueKind::Enum(allowed) => {
            if allowed.contains(&raw) {
                toml::Value::String(raw.to_owned())
            } else {
                return Err(crate::Error::Config(format!(
                    "{path} must be one of {}",
                    allowed.join(", ")
                )));
            }
        }
    };
    Ok(value)
}

/// Return the error used for a path that is not an editable/known leaf.
#[must_use]
pub fn unknown_key(path: &str) -> crate::Error {
    crate::Error::Config(format!(
        "unknown or non-editable configuration key `{path}`"
    ))
}

/// Validate semantic relationships that serde cannot express.
pub fn validate_settings(settings: &Settings) -> crate::Result<()> {
    if settings.version != super::CONFIG_VERSION {
        return Err(crate::Error::Config(format!(
            "unsupported configuration version {}; expected {}",
            settings.version,
            super::CONFIG_VERSION
        )));
    }
    if settings.embedding.batch_size == 0 {
        return Err(config_error("embedding.batch_size", "must be positive"));
    }
    if settings.embedding.model_dimension == Some(0) {
        return Err(config_error(
            "embedding.model_dimension",
            "must be positive",
        ));
    }
    if settings.embedding.output_dimension == Some(0) {
        return Err(config_error(
            "embedding.output_dimension",
            "must be positive",
        ));
    }
    if let (Some(model), Some(output)) = (
        settings.embedding.model_dimension,
        settings.embedding.output_dimension,
    ) && output > model
    {
        return Err(config_error(
            "embedding.output_dimension",
            "must not exceed embedding.model_dimension",
        ));
    }

    validate_http(
        "embedding.openai_compatible",
        &settings.embedding.openai_compatible.http,
    )?;
    validate_http("embedding.tei_native", &settings.embedding.tei_native.http)?;
    validate_http("embedding.ollama", &settings.embedding.ollama.http)?;
    if !TRUNCATION_DIRECTIONS.contains(&settings.embedding.tei_native.truncation_direction.as_str())
    {
        return Err(config_error(
            "embedding.tei_native.truncation_direction",
            "must be left or right",
        ));
    }

    if settings.retrieval.top_k == 0 {
        return Err(config_error("retrieval.top_k", "must be positive"));
    }
    if !settings.retrieval.norm.softmax.temperature.is_finite()
        || settings.retrieval.norm.softmax.temperature <= 0.0
    {
        return Err(config_error(
            "retrieval.norm.softmax.temperature",
            "must be finite and positive",
        ));
    }
    let linear = &settings.retrieval.fusion.linear;
    if !linear.semantic_factor.is_finite() || !linear.lexical_factor.is_finite() {
        return Err(config_error(
            "retrieval.fusion.linear",
            "factors must be finite",
        ));
    }
    if linear.semantic_factor == 0.0 && linear.lexical_factor == 0.0 {
        return Err(config_error(
            "retrieval.fusion.linear",
            "semantic_factor and lexical_factor must not both be zero",
        ));
    }
    if settings.retrieval.fusion.rrf.k == 0 {
        return Err(config_error("retrieval.fusion.rrf.k", "must be positive"));
    }

    if settings.chunking.size == 0 {
        return Err(config_error("chunking.size", "must be positive"));
    }
    if settings.chunking.overlap >= settings.chunking.size {
        return Err(config_error(
            "chunking.overlap",
            "must be less than chunking.size",
        ));
    }
    if !FTS_STRATEGIES.contains(&settings.fts.strategy.as_str()) {
        return Err(config_error(
            "fts.strategy",
            "must be unicode61, trigram, simple, or jieba",
        ));
    }
    Ok(())
}

fn validate_http(path: &str, config: &EmbeddingHttpConfig) -> crate::Result<()> {
    let timeout_path = format!("{path}.timeout");
    if !config.timeout.is_finite() || config.timeout <= 0.0 {
        return Err(config_error(&timeout_path, "must be finite and positive"));
    }
    if std::time::Duration::try_from_secs_f64(config.timeout).is_err() {
        return Err(config_error(
            &timeout_path,
            "is outside the supported duration range",
        ));
    }
    Ok(())
}

fn config_error(path: &str, message: &str) -> crate::Error {
    crate::Error::Config(format!("{path} {message}"))
}

fn parse_bool(path: &str, raw: &str) -> crate::Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(true),
        "false" | "0" | "no" | "n" | "off" => Ok(false),
        _ => Err(config_error(path, "must be true or false")),
    }
}

fn parse_integer<T>(path: &str, raw: &str) -> crate::Result<T>
where
    T: std::str::FromStr,
{
    raw.trim()
        .parse()
        .map_err(|_| config_error(path, "must be a non-negative integer"))
}

fn parse_float(path: &str, raw: &str) -> crate::Result<f64> {
    let value: f64 = raw
        .trim()
        .parse()
        .map_err(|_| config_error(path, "must be a number"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(config_error(path, "must be finite"))
    }
}

fn integer_error(path: &str) -> crate::Error {
    config_error(path, "is too large for TOML")
}

fn parse_string_list(path: &str, raw: &str) -> crate::Result<Vec<toml::Value>> {
    let trimmed = raw.trim();
    let values = if trimmed.starts_with('[') {
        serde_json::from_str::<Vec<String>>(trimmed).map_err(|_| {
            config_error(path, "must be a JSON string array or comma-separated list")
        })?
    } else if trimmed.is_empty() {
        Vec::new()
    } else {
        trimmed
            .split(',')
            .map(|item| item.trim().to_owned())
            .collect()
    };
    Ok(values.into_iter().map(toml::Value::String).collect())
}

/// Replace configured secrets in a JSON representation in place.
pub fn redact_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, value) in object {
                if key == "key" && !value.is_null() {
                    *value = serde_json::Value::String("[redacted]".to_string());
                } else {
                    redact_json(value);
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                redact_json(value);
            }
        }
        _ => {}
    }
}

/// Serialize effective settings for machine-readable output with secrets
/// redacted.
pub fn redacted_json(settings: &Settings) -> crate::Result<serde_json::Value> {
    let mut value = serde_json::to_value(settings)
        .map_err(|error| crate::Error::Config(format!("cannot serialize config: {error}")))?;
    redact_json(&mut value);
    Ok(value)
}

/// Serialize effective settings as TOML with secrets redacted.
pub fn redacted_toml(settings: &Settings) -> crate::Result<String> {
    let mut value = toml::Value::try_from(settings)
        .map_err(|error| crate::Error::Config(format!("cannot serialize config: {error}")))?;
    redact_toml(&mut value);
    toml::to_string_pretty(&value)
        .map_err(|error| crate::Error::Config(format!("cannot serialize config: {error}")))
}

fn redact_toml(value: &mut toml::Value) {
    match value {
        toml::Value::Table(table) => {
            for (key, value) in table {
                if key == "key" {
                    *value = toml::Value::String("[redacted]".to_string());
                } else {
                    redact_toml(value);
                }
            }
        }
        toml::Value::Array(values) => {
            for value in values {
                redact_toml(value);
            }
        }
        _ => {}
    }
}

/// Read a dotted path from a JSON object.
pub fn json_path<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for part in path.split('.') {
        current = current.get(part)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LinearFusionConfig, RetrievalConfig};

    #[test]
    fn parses_friendly_booleans_and_lists() {
        assert_eq!(
            parse_value("app.output.no_color", "YeS").unwrap(),
            toml::Value::Boolean(true)
        );
        assert_eq!(
            parse_value("app.ignore.sources", "[\".gitignore\", \"custom.ignore\"]").unwrap(),
            toml::Value::Array(vec![
                toml::Value::String(".gitignore".to_string()),
                toml::Value::String("custom.ignore".to_string()),
            ])
        );
    }

    #[test]
    fn rejects_old_normalization_switch_and_version_edit() {
        assert!(find_key("embedding.l2_normalize").is_none());
        assert!(!find_key("version").unwrap().editable);
        assert!(
            unknown_key("embedding.l2_normalize")
                .to_string()
                .contains("unknown")
        );
    }

    #[test]
    fn rejects_zero_linear_factors() {
        let settings = Settings {
            retrieval: RetrievalConfig {
                fusion: super::super::FusionConfig {
                    linear: LinearFusionConfig {
                        semantic_factor: 0.0,
                        lexical_factor: 0.0,
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_settings(&settings).is_err());
    }

    #[test]
    fn rejects_unsupported_version_and_invalid_chunking() {
        let settings = Settings {
            version: 2,
            ..Default::default()
        };
        assert!(
            validate_settings(&settings)
                .unwrap_err()
                .to_string()
                .contains("version")
        );

        let settings = Settings {
            chunking: super::super::ChunkingConfig {
                size: 128,
                overlap: 128,
            },
            ..Default::default()
        };
        assert!(
            validate_settings(&settings)
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );
    }

    #[test]
    fn parses_comma_lists_and_optional_strings_as_strings() {
        assert_eq!(
            parse_value("app.ignore.sources", " .gitignore, custom.ignore ").unwrap(),
            toml::Value::Array(vec![
                toml::Value::String(".gitignore".to_string()),
                toml::Value::String("custom.ignore".to_string()),
            ])
        );
        assert_eq!(
            parse_value("embedding.openai_compatible.key", "  secret  ").unwrap(),
            toml::Value::String("  secret  ".to_string())
        );
    }
}
