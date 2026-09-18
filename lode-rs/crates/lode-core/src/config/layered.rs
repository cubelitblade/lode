//! Layered configuration loading and editing.
//!
//! Sources ascend in precedence: defaults < user config < `.lode.toml` <
//! `lode.toml` < `.lode/config.toml` < `LODE_...` environment variables.
//! CLI-only overrides are applied by the CLI adapter after loading. TOML
//! editing deliberately lives here so every frontend uses the same key
//! schema, parsing, validation, and atomic-write behavior.

use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use toml_edit::{DocumentMut, InlineTable, Item, Value, value as edit_value};

use crate::config::Settings;
use crate::config::schema::{self, ConfigKey, ValueKind};

/// Relative project config filenames, lowest precedence first.
const PROJECT_CONFIG_PATHS: [&str; 3] = [".lode.toml", "lode.toml", ".lode/config.toml"];

/// Platform used by [`user_config_dir_from_env`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigPlatform {
    /// Linux and other XDG-style Unix environments.
    Linux,
    /// macOS application-support layout.
    Macos,
    /// Windows roaming application-data layout.
    Windows,
}

/// Return the platform-specific directory holding lode's user config.
pub fn user_config_dir_from_env(
    platform: ConfigPlatform,
    xdg_config_home: Option<&OsStr>,
    home: Option<&OsStr>,
    appdata: Option<&OsStr>,
) -> crate::Result<PathBuf> {
    if let Some(dir) = xdg_config_home.filter(|dir| !dir.is_empty()) {
        return Ok(PathBuf::from(dir).join("lode"));
    }
    match platform {
        ConfigPlatform::Linux => home
            .filter(|dir| !dir.is_empty())
            .map(|dir| PathBuf::from(dir).join(".config").join("lode"))
            .ok_or_else(|| missing_user_dir("HOME")),
        ConfigPlatform::Macos => home
            .filter(|dir| !dir.is_empty())
            .map(|dir| PathBuf::from(dir).join("Library/Application Support/lode"))
            .ok_or_else(|| missing_user_dir("HOME")),
        ConfigPlatform::Windows => appdata
            .filter(|dir| !dir.is_empty())
            .map(|dir| PathBuf::from(dir).join("lode"))
            .ok_or_else(|| missing_user_dir("APPDATA")),
    }
}

fn missing_user_dir(variable: &str) -> crate::Error {
    crate::Error::Config(format!(
        "could not determine the user configuration directory; set {variable}"
    ))
}

fn current_platform() -> ConfigPlatform {
    if cfg!(target_os = "windows") {
        ConfigPlatform::Windows
    } else if cfg!(target_os = "macos") {
        ConfigPlatform::Macos
    } else {
        ConfigPlatform::Linux
    }
}

/// Directory holding the user-level config file.
pub fn user_config_dir() -> crate::Result<PathBuf> {
    user_config_dir_from_env(
        current_platform(),
        std::env::var_os("XDG_CONFIG_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
        std::env::var_os("APPDATA").as_deref(),
    )
}

/// Path of the user-level config file.
pub fn user_config_path() -> crate::Result<PathBuf> {
    Ok(user_config_dir()?.join("config.toml"))
}

/// Path of the highest-precedence project config file.
pub fn workspace_config_path(base: &Path) -> PathBuf {
    PROJECT_CONFIG_PATHS
        .iter()
        .rev()
        .map(|name| base.join(name))
        .find(|path| path.is_file())
        .unwrap_or_else(|| base.join(".lode/config.toml"))
}

/// Candidate config files in ascending precedence (user, then project).
fn discover_config_files(base: &Path, user_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![user_dir.join("config.toml")];
    for name in PROJECT_CONFIG_PATHS {
        candidates.push(base.join(name));
    }
    candidates
        .into_iter()
        .filter(|path| path.is_file())
        .collect()
}

/// Merge a TOML table into `acc`, overriding scalars and recursing into equal
/// named tables.
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

/// Parse a TOML file, treating malformed files as an error.
fn parse_file(path: &Path) -> crate::Result<toml::value::Table> {
    let text = fs::read_to_string(path)?;
    let value = text.parse::<toml::Value>().map_err(|error| {
        toml_parse_error(
            &format!("invalid TOML in {}", path.display()),
            &text,
            error.message(),
            error.span(),
        )
    })?;
    value.as_table().cloned().ok_or_else(|| {
        crate::Error::Config(format!(
            "expected a TOML table at top level in {}",
            path.display()
        ))
    })
}

/// Format a TOML error without including the source line, which may contain
/// a provider API key or another secret value.
fn toml_parse_error(
    context: &str,
    source: &str,
    message: &str,
    span: Option<std::ops::Range<usize>>,
) -> crate::Error {
    let location = span.map_or_else(String::new, |span| {
        let mut offset = span.start.min(source.len());
        while offset > 0 && !source.is_char_boundary(offset) {
            offset -= 1;
        }
        let prefix = &source[..offset];
        let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
        let column = prefix[line_start..].chars().count() + 1;
        format!(" at line {line}, column {column}")
    });
    crate::Error::Config(format!("{context}{location}: {message}"))
}

fn get_toml_path<'a>(table: &'a toml::value::Table, path: &str) -> Option<&'a toml::Value> {
    let mut current = table;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        let value = current.get(part)?;
        if parts.peek().is_some() {
            current = value.as_table()?;
        } else {
            return Some(value);
        }
    }
    None
}

fn set_toml_path(
    table: &mut toml::value::Table,
    path: &str,
    new_value: toml::Value,
) -> crate::Result<()> {
    let mut current = table;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            current.insert(part.to_owned(), new_value);
            return Ok(());
        }
        let child = current
            .entry(part.to_owned())
            .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
        current = child.as_table_mut().ok_or_else(|| {
            crate::Error::Config(format!("configuration path `{path}` contains a non-table"))
        })?;
    }
    Ok(())
}

/// Apply a snapshot of environment variables to a merged TOML table.
///
/// Unknown variables are intentionally ignored, including the removed
/// `LODE_EMBEDDING__L2_NORMALIZE` compatibility variable. Known variables are
/// parsed by the same schema used by `config set`.
fn apply_environment(
    acc: &mut toml::value::Table,
    environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> crate::Result<()> {
    for (name, raw_value) in environment {
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(path) = name.strip_prefix("LODE_") else {
            continue;
        };
        let path = path
            .split("__")
            .filter(|part| !part.is_empty())
            .map(str::to_ascii_lowercase)
            .collect::<Vec<_>>()
            .join(".");
        let Some(key) = schema::find_key(&path) else {
            continue;
        };
        let raw_value = raw_value
            .to_str()
            .ok_or_else(|| crate::Error::Config(format!("{name} must contain valid UTF-8")))?;
        let parsed = schema::parse_value(&path, raw_value)?;
        set_toml_path(acc, key.path, parsed)?;
    }
    Ok(())
}

fn settings_from_table(table: toml::value::Table) -> crate::Result<Settings> {
    // Deserialize on top of the complete defaults so a partial nested table
    // retains provider-specific defaults such as the OpenAI endpoint.
    let mut default_value = toml::Value::try_from(Settings::default())
        .map_err(|error| crate::Error::Config(format!("cannot build default config: {error}")))?;
    let defaults = default_value.as_table_mut().ok_or_else(|| {
        crate::Error::Config("default configuration is not a TOML table".to_string())
    })?;
    merge(defaults, &table);
    let settings: Settings = default_value.try_into().map_err(|_error| {
        crate::Error::Config(
            "cannot interpret config: one or more values have an invalid type or value".to_string(),
        )
    })?;
    schema::validate_settings(&settings)?;
    Ok(settings)
}

/// Load settings from explicit roots using the process environment.
fn load_settings_from(base: &Path, user_dir: &Path) -> crate::Result<Settings> {
    load_settings_from_with_env(base, user_dir, std::env::vars_os())
}

fn load_settings_from_with_env(
    base: &Path,
    user_dir: &Path,
    environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> crate::Result<Settings> {
    let mut acc = toml::value::Table::new();
    for path in discover_config_files(base, user_dir) {
        merge(&mut acc, &parse_file(&path)?);
    }
    apply_environment(&mut acc, environment)?;
    settings_from_table(acc)
}

/// Load settings following the configured precedence chain.
pub fn load_settings_for(base: &Path) -> crate::Result<Settings> {
    load_settings_from(base, &user_config_dir()?)
}

/// Convenience overload rooting discovery at the current directory.
pub fn load_settings() -> crate::Result<Settings> {
    load_settings_for(Path::new("."))
}

/// Read one explicit value from one config file without loading any other
/// layer. Missing files and keys are represented by `None`.
pub fn get_explicit_value(path: &Path, key: &str) -> crate::Result<Option<toml::Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let table = parse_file(path)?;
    Ok(get_toml_path(&table, key).cloned())
}

/// Validate a candidate TOML document without applying environment variables.
fn validate_document(document: &DocumentMut) -> crate::Result<()> {
    let text = document.to_string();
    let value = text.parse::<toml::Value>().map_err(|error| {
        toml_parse_error("cannot validate TOML", &text, error.message(), error.span())
    })?;
    let table = value
        .as_table()
        .cloned()
        .ok_or_else(|| crate::Error::Config("expected a TOML table".to_string()))?;
    settings_from_table(table).map(|_| ())
}

fn edit_item(value: toml::Value) -> crate::Result<Item> {
    match value {
        toml::Value::String(value) => Ok(edit_value(value)),
        toml::Value::Integer(value) => Ok(edit_value(value)),
        toml::Value::Float(value) => Ok(edit_value(value)),
        toml::Value::Boolean(value) => Ok(edit_value(value)),
        toml::Value::Array(values) => {
            let mut array = toml_edit::Array::new();
            for value in values {
                match value {
                    toml::Value::String(value) => array.push(value),
                    _ => {
                        return Err(crate::Error::Config(
                            "configuration string lists may contain only strings".to_string(),
                        ));
                    }
                }
            }
            Ok(edit_value(array))
        }
        toml::Value::Datetime(_) | toml::Value::Table(_) => Err(crate::Error::Config(
            "configuration values must be scalar or string-list values".to_string(),
        )),
    }
}

fn set_edit_path(
    document: &mut DocumentMut,
    key: &ConfigKey,
    new_value: toml::Value,
) -> crate::Result<()> {
    let item = edit_item(new_value)?;
    let mut parts = key.path.split('.').peekable();
    set_edit_item(document.as_item_mut(), &mut parts, item, key.path)
}

fn set_edit_item<'a, I>(
    item: &mut Item,
    parts: &mut std::iter::Peekable<I>,
    mut value: Item,
    path: &str,
) -> crate::Result<()>
where
    I: Iterator<Item = &'a str>,
{
    if let Item::Value(Value::InlineTable(table)) = item {
        return set_edit_inline_table(table, parts, value, path);
    }
    let part = parts
        .next()
        .ok_or_else(|| crate::Error::Config("empty configuration key".to_string()))?;
    let table = item.as_table_like_mut().ok_or_else(|| {
        crate::Error::Config(format!("configuration path `{path}` contains a non-table"))
    })?;
    if parts.peek().is_none() {
        let old_decor = table
            .get(part)
            .and_then(Item::as_value)
            .map(|old| old.decor().clone());
        if let (Some(new), Some(old_decor)) = (value.as_value_mut(), old_decor) {
            *new.decor_mut() = old_decor;
        }
        table.insert(part, value);
        return Ok(());
    }
    let child = table.entry(part).or_insert_with(toml_edit::table);
    set_edit_item(child, parts, value, path)
}

fn set_edit_inline_table<'a, I>(
    table: &mut InlineTable,
    parts: &mut std::iter::Peekable<I>,
    value: Item,
    path: &str,
) -> crate::Result<()>
where
    I: Iterator<Item = &'a str>,
{
    let part = parts
        .next()
        .ok_or_else(|| crate::Error::Config("empty configuration key".to_string()))?;
    if parts.peek().is_none() {
        let mut value = value.into_value().map_err(|_| {
            crate::Error::Config(
                "configuration values must be scalar or string-list values".to_string(),
            )
        })?;
        if let Some(old) = table.get(part) {
            *value.decor_mut() = old.decor().clone();
        }
        table.insert(part, value);
        return Ok(());
    }
    let child = table.get_or_insert(part, InlineTable::new());
    let child = child.as_inline_table_mut().ok_or_else(|| {
        crate::Error::Config(format!("configuration path `{path}` contains a non-table"))
    })?;
    set_edit_inline_table(child, parts, value, path)
}

fn remove_edit_path(document: &mut DocumentMut, key: &ConfigKey) -> bool {
    let mut parts = key.path.split('.').peekable();
    remove_edit_item(document.as_item_mut(), &mut parts)
}

fn remove_edit_item<'a, I>(item: &mut Item, parts: &mut std::iter::Peekable<I>) -> bool
where
    I: Iterator<Item = &'a str>,
{
    let Some(part) = parts.next() else {
        return false;
    };
    let Some(table) = item.as_table_like_mut() else {
        return false;
    };
    if parts.peek().is_none() {
        return table.remove(part).is_some();
    }
    table
        .get_mut(part)
        .is_some_and(|child| remove_edit_item(child, parts))
}

fn read_document(path: &Path) -> crate::Result<DocumentMut> {
    if !path.exists() {
        return Ok(DocumentMut::new());
    }
    let text = fs::read_to_string(path)?;
    text.parse::<DocumentMut>().map_err(|error| {
        toml_parse_error(
            &format!("invalid TOML in {}", path.display()),
            &text,
            error.message(),
            error.span(),
        )
    })
}

#[cfg(not(windows))]
fn atomic_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::rename(from, to)
}

#[cfg(windows)]
fn atomic_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    #[allow(non_snake_case)]
    unsafe extern "system" {
        fn MoveFileExW(existing_name: *const u16, new_name: *const u16, flags: u32) -> i32;
    }

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let from = from
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let to = to
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // MoveFileExW replaces an existing destination and keeps the temporary
    // file in the same directory, so the visible update remains atomic.
    let replaced = unsafe {
        MoveFileExW(
            from.as_ptr(),
            to.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn atomic_write(path: &Path, contents: &str) -> crate::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let existing_permissions = match fs::metadata(path) {
        Ok(metadata) => Some(metadata.permissions()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    fs::create_dir_all(parent)?;
    let file_name = path.file_name().and_then(OsStr::to_str).ok_or_else(|| {
        crate::Error::Config(format!(
            "configuration path has no valid filename: {}",
            path.display()
        ))
    })?;
    let mut temporary_path = None;
    let mut temporary_file = None;
    for attempt in 0..100_u32 {
        let candidate = parent.join(format!(".{file_name}.tmp-{}-{attempt}", std::process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&candidate) {
            Ok(file) => {
                temporary_path = Some(candidate);
                temporary_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let Some(temporary_path) = temporary_path else {
        return Err(crate::Error::Config(format!(
            "could not create a temporary config file beside {}",
            path.display()
        )));
    };
    let mut file = temporary_file.expect("temporary path has a file");
    if let Err(error) = file.write_all(contents.as_bytes()) {
        let _ = fs::remove_file(&temporary_path);
        return Err(error.into());
    }
    if let Some(permissions) = existing_permissions
        && let Err(error) = file.set_permissions(permissions)
    {
        let _ = fs::remove_file(&temporary_path);
        return Err(error.into());
    }
    if let Err(error) = file.sync_all() {
        let _ = fs::remove_file(&temporary_path);
        return Err(error.into());
    }
    drop(file);
    if let Err(error) = atomic_replace(&temporary_path, path) {
        let _ = fs::remove_file(&temporary_path);
        return Err(error.into());
    }
    Ok(())
}

/// Set an editable key in a selected config file and atomically replace it.
pub fn set_value(path: &Path, key: &str, new_value: toml::Value) -> crate::Result<bool> {
    let key = schema::find_key(key)
        .filter(|key| key.editable)
        .ok_or_else(|| schema::unknown_key(key))?;
    let mut document = read_document(path)?;
    let old = get_explicit_value_from_document(&document, key.path);
    set_edit_path(&mut document, key, new_value)?;
    validate_document(&document)?;
    let new = get_explicit_value_from_document(&document, key.path);
    let changed = old != new;
    if changed {
        atomic_write(path, &document.to_string())?;
    }
    Ok(changed)
}

/// Remove an editable key from a selected config file and atomically replace
/// it. Missing explicit keys are reported as errors.
pub fn unset_value(path: &Path, key: &str) -> crate::Result<bool> {
    let key = schema::find_key(key)
        .filter(|key| key.editable)
        .ok_or_else(|| schema::unknown_key(key))?;
    let mut document = read_document(path)?;
    if !remove_edit_path(&mut document, key) {
        return Err(crate::Error::Config(format!(
            "configuration key `{}` is not set",
            key.path
        )));
    }
    validate_document(&document)?;
    atomic_write(path, &document.to_string())?;
    Ok(true)
}

fn get_explicit_value_from_document(document: &DocumentMut, key: &str) -> Option<toml::Value> {
    let table = document
        .to_string()
        .parse::<toml::Value>()
        .ok()?
        .as_table()?
        .clone();
    get_toml_path(&table, key).cloned()
}

/// Read a scoped config file as a TOML table.
pub fn read_explicit_table(path: &Path) -> crate::Result<toml::value::Table> {
    if !path.is_file() {
        return Ok(toml::value::Table::new());
    }
    parse_file(path)
}

/// Convert a schema value kind to a human-readable type description.
#[must_use]
pub const fn value_kind_name(kind: ValueKind) -> &'static str {
    match kind {
        ValueKind::Bool => "boolean",
        ValueKind::String | ValueKind::OptionalString => "string",
        ValueKind::U32 | ValueKind::Usize | ValueKind::U64 => "integer",
        ValueKind::Float => "number",
        ValueKind::StringList => "string list",
        ValueKind::Enum(_) => "enum",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmbeddingProvider, OutputPalette};

    fn load_without_env(base: &Path, user: &Path) -> crate::Result<Settings> {
        load_settings_from_with_env(base, user, std::iter::empty())
    }

    #[test]
    fn defaults_when_no_config_exists() {
        let dir = tempfile::tempdir().unwrap();
        let settings = load_without_env(dir.path(), dir.path()).unwrap();
        assert_eq!(settings.embedding.model_dimension, None);
        assert_eq!(settings.chunking.size, 1024);
        assert_eq!(settings.chunking.overlap, 128);
        assert_eq!(settings.fts.strategy, "simple");
        assert!(settings.app.ignore.sources.is_empty());
    }

    #[test]
    fn reads_model_dimension_from_project_toml() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("lode.toml"),
            "[embedding]\nmodel_dimension = 768\n",
        )
        .unwrap();
        let settings = load_without_env(dir.path(), dir.path()).unwrap();
        assert_eq!(settings.embedding.model_dimension, Some(768));
    }

    #[test]
    fn lower_precedence_user_is_overridden_by_project() {
        let dir = tempfile::tempdir().unwrap();
        let user_dir = dir.path().join("user");
        fs::create_dir_all(&user_dir).unwrap();
        fs::write(
            user_dir.join("config.toml"),
            "[embedding]\nmodel_dimension = 384\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("lode.toml"),
            "[embedding]\nmodel_dimension = 640\n",
        )
        .unwrap();
        let settings = load_without_env(dir.path(), &user_dir).unwrap();
        assert_eq!(settings.embedding.model_dimension, Some(640));
    }

    #[test]
    fn later_project_file_wins() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(".lode.toml"), "[chunking]\nsize = 512\n").unwrap();
        fs::write(dir.path().join("lode.toml"), "[chunking]\nsize = 2048\n").unwrap();
        let settings = load_without_env(dir.path(), dir.path()).unwrap();
        assert_eq!(settings.chunking.size, 2048);
    }

    #[test]
    fn rejects_malformed_toml() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("lode.toml"), "[[[\n").unwrap();
        let err = load_without_env(dir.path(), dir.path()).unwrap_err();
        assert!(err.to_string().contains("invalid TOML"));
    }

    #[test]
    fn reads_output_config_from_project_toml() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("lode.toml"),
            "[app.output]\npalette = \"accessible_okabe_ito\"\nno_color = true\n",
        )
        .unwrap();
        let settings = load_without_env(dir.path(), dir.path()).unwrap();
        assert_eq!(
            settings.app.output.palette,
            OutputPalette::AccessibleOkabeIto
        );
        assert_eq!(settings.app.output.no_color, Some(true));
    }

    #[test]
    fn rejects_invalid_output_palette() {
        let dir = tempfile::tempdir().unwrap();
        for palette in ["neon", "accessible_light", "accessible_dark"] {
            fs::write(
                dir.path().join("lode.toml"),
                format!("[app.output]\npalette = \"{palette}\"\n"),
            )
            .unwrap();
            let error = load_without_env(dir.path(), dir.path()).unwrap_err();
            assert!(error.to_string().contains("cannot interpret config"));
            assert!(!error.to_string().contains(palette));
        }
    }

    #[test]
    fn environment_overrides_toml_and_ignores_unknown_values() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("lode.toml"), "[chunking]\nsize = 512\n").unwrap();
        let environment = vec![
            (
                OsString::from("LODE_CHUNKING__SIZE"),
                OsString::from("2048"),
            ),
            (
                OsString::from("LODE_EMBEDDING__L2_NORMALIZE"),
                OsString::from("false"),
            ),
            (OsString::from("LODE_UNKNOWN__VALUE"), OsString::from("bad")),
        ];
        let settings = load_settings_from_with_env(dir.path(), dir.path(), environment).unwrap();
        assert_eq!(settings.chunking.size, 2048);
    }

    #[test]
    fn output_environment_uses_the_shared_parser() {
        let dir = tempfile::tempdir().unwrap();
        let environment = vec![
            (
                OsString::from("LODE_APP__OUTPUT__PALETTE"),
                OsString::from("accessible_tol"),
            ),
            (
                OsString::from("LODE_APP__OUTPUT__NO_COLOR"),
                OsString::from("off"),
            ),
        ];
        let settings = load_settings_from_with_env(dir.path(), dir.path(), environment).unwrap();
        assert_eq!(settings.app.output.palette, OutputPalette::AccessibleTol);
        assert_eq!(settings.app.output.no_color, Some(false));
    }

    #[test]
    fn partial_provider_override_keeps_provider_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let environment = vec![(
            OsString::from("LODE_EMBEDDING__OPENAI_COMPATIBLE__KEY"),
            OsString::from("secret"),
        )];
        let settings = load_settings_from_with_env(dir.path(), dir.path(), environment).unwrap();
        assert_eq!(
            settings.embedding.openai_compatible.http.endpoint,
            "https://api.openai.com/v1"
        );
        assert_eq!(
            settings.embedding.openai_compatible.http.key.as_deref(),
            Some("secret")
        );
    }

    #[test]
    fn user_config_paths_follow_each_platform() {
        assert_eq!(
            user_config_dir_from_env(
                ConfigPlatform::Linux,
                None,
                Some(OsStr::new("/home/u")),
                None,
            )
            .unwrap(),
            PathBuf::from("/home/u/.config/lode")
        );
        assert_eq!(
            user_config_dir_from_env(
                ConfigPlatform::Macos,
                None,
                Some(OsStr::new("/Users/u")),
                None,
            )
            .unwrap(),
            PathBuf::from("/Users/u/Library/Application Support/lode")
        );
        assert_eq!(
            user_config_dir_from_env(
                ConfigPlatform::Windows,
                None,
                None,
                Some(OsStr::new("C:\\Users\\u\\AppData\\Roaming")),
            )
            .unwrap(),
            PathBuf::from("C:\\Users\\u\\AppData\\Roaming/lode")
        );
        assert_eq!(
            user_config_dir_from_env(
                ConfigPlatform::Linux,
                Some(OsStr::new("/tmp/config")),
                None,
                None,
            )
            .unwrap(),
            PathBuf::from("/tmp/config/lode")
        );
    }

    #[test]
    fn scoped_edit_preserves_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lode.toml");
        fs::write(
            &path,
            "# keep this\n[embedding]\nmodel = \"demo\"\n[chunking]\nsize = 512 # keep inline\n",
        )
        .unwrap();
        assert!(set_value(&path, "chunking.size", toml::Value::Integer(2048)).unwrap());
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("# keep this"));
        assert!(text.contains("# keep inline"));
        assert!(text.contains("size = 2048"));
        assert!(unset_value(&path, "chunking.size").unwrap());
        assert!(!fs::read_to_string(&path).unwrap().contains("size = 2048"));
    }

    #[cfg(unix)]
    #[test]
    fn scoped_edit_preserves_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lode.toml");
        fs::write(&path, "[chunking]\nsize = 512\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        set_value(&path, "chunking.size", toml::Value::Integer(2048)).unwrap();

        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn embedding_defaults_remain_compatible() {
        let dir = tempfile::tempdir().unwrap();
        let settings = load_without_env(dir.path(), dir.path()).unwrap();
        assert_eq!(
            settings.embedding.provider,
            EmbeddingProvider::OpenAICompatible
        );
        assert_eq!(settings.embedding.batch_size, 32);
    }
}
