use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;
use tempfile::TempDir;

fn run_lode(workspace: &Path, config_home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_lode"))
        .arg("--workspace")
        .arg(workspace)
        .arg("--view")
        .arg("json")
        .args(args)
        .env("XDG_CONFIG_HOME", config_home)
        .env_remove("NO_COLOR")
        .env_remove("LODE_APP__OUTPUT__PALETTE")
        .env_remove("LODE_APP__OUTPUT__NO_COLOR")
        .output()
        .expect("failed to start lode")
}

fn json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON ({error}); stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn show_is_redacted_and_drops_the_removed_normalization_field() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    fs::write(
        workspace.path().join("lode.toml"),
        "[embedding]\nl2_normalize = false\n[embedding.openai_compatible]\nkey = \"super-secret\"\n[fts]\nstrategy = \"unicode61\"\n",
    )
    .unwrap();

    let output = run_lode(workspace.path(), config_home.path(), &["config", "show"]);
    assert!(output.status.success());
    let payload = json(&output);
    assert_eq!(
        payload["embedding"]["openai_compatible"]["key"],
        "[redacted]"
    );
    assert!(payload["embedding"].get("l2_normalize").is_none());
}

#[test]
fn output_flags_override_effective_configuration() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    fs::write(
        workspace.path().join("lode.toml"),
        "[app.output]\npalette = \"ansi\"\nno_color = false\n",
    )
    .unwrap();

    let output = run_lode(
        workspace.path(),
        config_home.path(),
        &[
            "--palette",
            "accessible_tol",
            "--no-color",
            "config",
            "show",
        ],
    );

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload = json(&output);
    assert_eq!(payload["app"]["output"]["palette"], "accessible_tol");
    assert_eq!(payload["app"]["output"]["no_color"], true);
}

#[test]
fn set_get_unset_uses_the_highest_existing_workspace_file() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let config = workspace.path().join("lode.toml");
    fs::write(&config, "# keep this comment\n[chunking]\nsize = 512\n").unwrap();

    let set = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "set", "chunking.size", "2048"],
    );
    assert!(set.status.success());
    assert!(String::from_utf8_lossy(&fs::read(&config).unwrap()).contains("# keep this comment"));
    assert!(String::from_utf8_lossy(&fs::read(&config).unwrap()).contains("size = 2048"));
    assert_eq!(json(&set)["changed"], true);
    assert!(json(&set)["path"].as_str().unwrap().ends_with("lode.toml"));

    let get = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "get", "chunking.size", "--scope", "workspace"],
    );
    assert!(get.status.success());
    assert_eq!(json(&get)["value"], 2048);

    let unset = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "unset", "chunking.size"],
    );
    assert!(unset.status.success());
    assert!(!String::from_utf8_lossy(&fs::read(&config).unwrap()).contains("size = 2048"));

    let effective = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "get", "chunking.size"],
    );
    assert!(effective.status.success());
    assert_eq!(json(&effective)["value"], 1024);
}

#[test]
fn set_handles_inline_tables_with_missing_nested_paths() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let config = workspace.path().join("lode.toml");
    fs::write(
        &config,
        "# Keep this configuration note\napp = { output = { palette = \"ansi\" } } # Keep this inline note\n",
    )
    .unwrap();

    let set = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "set", "app.ignore.sources", "[\".gitignore\"]"],
    );
    assert!(set.status.success());
    assert_eq!(json(&set)["changed"], true);
    let text = fs::read_to_string(&config).unwrap();
    assert!(text.contains("# Keep this configuration note"));
    assert!(text.contains("# Keep this inline note"));

    let get = run_lode(
        workspace.path(),
        config_home.path(),
        &[
            "config",
            "get",
            "app.ignore.sources",
            "--scope",
            "workspace",
        ],
    );
    assert!(get.status.success());
    assert_eq!(json(&get)["value"], serde_json::json!([".gitignore"]));
}

#[test]
fn user_scope_is_isolated_from_workspace_scope() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();

    let set = run_lode(
        workspace.path(),
        config_home.path(),
        &[
            "config",
            "set",
            "embedding.model",
            "user-model",
            "--scope",
            "user",
        ],
    );
    assert!(set.status.success());
    assert!(config_home.path().join("lode/config.toml").is_file());

    let workspace_get = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "get", "embedding.model", "--scope", "workspace"],
    );
    assert_eq!(workspace_get.status.code(), Some(1));

    let effective = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "get", "embedding.model"],
    );
    assert!(effective.status.success());
    assert_eq!(json(&effective)["value"], "user-model");
}

#[test]
fn invalid_edit_leaves_the_original_file_unchanged() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let config = workspace.path().join("lode.toml");
    fs::write(&config, "[chunking]\nsize = 512\noverlap = 128\n").unwrap();
    let before = fs::read_to_string(&config).unwrap();

    let output = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "set", "chunking.overlap", "512"],
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(fs::read_to_string(&config).unwrap(), before);
}

#[test]
fn rejects_timeout_outside_duration_range_without_editing() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    let config = workspace.path().join("lode.toml");
    fs::write(&config, "[embedding.openai_compatible]\ntimeout = 10.0\n").unwrap();
    let before = fs::read_to_string(&config).unwrap();

    let output = run_lode(
        workspace.path(),
        config_home.path(),
        &[
            "config",
            "set",
            "embedding.openai_compatible.timeout",
            "1e30",
        ],
    );

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(fs::read_to_string(&config).unwrap(), before);
    assert!(!String::from_utf8_lossy(&output.stdout).contains("panicked"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn malformed_toml_does_not_echo_secrets() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    fs::write(
        workspace.path().join("lode.toml"),
        "[embedding.openai_compatible]\nkey = \"super-secret\" invalid\n",
    )
    .unwrap();

    let output = run_lode(
        workspace.path(),
        config_home.path(),
        &["config", "set", "embedding.model", "demo"],
    );
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stdout.contains("super-secret"));
    assert!(!stderr.contains("super-secret"));
}

#[test]
fn invalid_sensitive_type_does_not_echo_values() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    fs::write(
        workspace.path().join("lode.toml"),
        "[embedding.openai_compatible]\nkey = 123456789012345\n",
    )
    .unwrap();

    let output = run_lode(workspace.path(), config_home.path(), &["config", "show"]);
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stdout.contains("123456789012345"));
    assert!(!stderr.contains("123456789012345"));
}

#[test]
fn negative_retrieval_factor_is_accepted_by_config_set() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();

    let output = run_lode(
        workspace.path(),
        config_home.path(),
        &[
            "config",
            "set",
            "retrieval.fusion.linear.semantic_factor",
            "-0.5",
        ],
    );
    assert!(
        output.status.success(),
        "set failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        fs::read_to_string(workspace.path().join(".lode/config.toml"))
            .unwrap()
            .contains("semantic_factor = -0.5")
    );
}

#[test]
fn configured_ignore_sources_affect_survey_and_old_env_is_ignored() {
    let workspace = TempDir::new().unwrap();
    let config_home = TempDir::new().unwrap();
    fs::write(workspace.path().join("extra.ignore"), "ignored.txt\n").unwrap();
    fs::write(workspace.path().join("ignored.txt"), "ignored").unwrap();
    fs::write(workspace.path().join("kept.txt"), "kept").unwrap();
    fs::write(
        workspace.path().join("lode.toml"),
        "[app.ignore]\nsources = [\"extra.ignore\"]\n[fts]\nstrategy = \"unicode61\"\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_lode"))
        .args([
            "--workspace",
            workspace.path().to_str().unwrap(),
            "--view",
            "json",
            "survey",
        ])
        .env("XDG_CONFIG_HOME", config_home.path())
        .env("LODE_EMBEDDING__L2_NORMALIZE", "false")
        .env_remove("NO_COLOR")
        .output()
        .unwrap();
    assert!(output.status.success());
    let payload = json(&output);
    let new_paths = payload["new"].as_array().unwrap();
    assert!(new_paths.iter().any(|path| path == "kept.txt"));
    assert!(!new_paths.iter().any(|path| path == "ignored.txt"));
}
