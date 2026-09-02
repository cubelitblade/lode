//! Layered config loading: combine defaults, TOML, env and kwargs.

use crate::config::Settings;

/// Load settings following the configured precedence chain.
pub fn load_settings() -> Result<Settings, crate::Error> {
    // Minimal scaffold: return defaults for now. Layering arrives next.
    Ok(Settings::default())
}
