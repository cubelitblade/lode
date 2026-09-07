//! Layered configuration.
//!
//! Precedence: `defaults < TOML < environment variables < constructor kwargs`.
//! The TOML source is treated as replaceable for future formats.

pub mod layered;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::embeddings::base::Embedder;
use crate::embeddings::http::{HttpClient, ReqwestHttpClient};
use crate::embeddings::ollama::OllamaBuilder;
use crate::embeddings::openai_compatible::OpenAiCompatibleBuilder;
use crate::embeddings::tei_native::TeiNativeBuilder;
use crate::index::records::Source;

/// Current config schema version; config files may declare `version`, and a
/// mismatch is refused at load time. Multi-format configs (future YAML) key
/// off this anchor.
pub const CONFIG_VERSION: u32 = 1;

/// Top-level application settings model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Config schema version declared by the file; defaults to the current
    /// version so legacy files without `version` keep loading.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Output / ignore switch configuration.
    #[serde(default)]
    pub app: AppConfig,
    /// Embedding provider configuration.
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    /// Retrieval configuration.
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    /// Chunking configuration.
    #[serde(default)]
    pub chunking: ChunkingConfig,
    /// FTS configuration.
    #[serde(default)]
    pub fts: FtsConfig,
    /// Ignore rules.
    #[serde(default)]
    pub ignore: IgnoreConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            app: AppConfig::default(),
            embedding: EmbeddingConfig::default(),
            retrieval: RetrievalConfig::default(),
            chunking: ChunkingConfig::default(),
            fts: FtsConfig::default(),
            ignore: IgnoreConfig::default(),
        }
    }
}

/// Default version for files that omit it (matches Python: current version).
fn default_version() -> u32 {
    CONFIG_VERSION
}

/// App-level configuration (output, ignore).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {}

/// Embedding backend provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProvider {
    /// Any OpenAI-compatible embeddings endpoint (TEI, Ollama, vLLM, hosted).
    #[default]
    OpenAICompatible,
    /// Hugging Face Text Embeddings Inference native API.
    TeiNative,
    /// Ollama native API.
    Ollama,
}

/// Transport settings shared by HTTP embedding backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingHttpConfig {
    /// Base URL of the embedding endpoint.
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    /// Optional bearer token sent as `Authorization: Bearer <key>`.
    #[serde(default)]
    pub key: Option<String>,
    /// Retry count for transient failures.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Per-request timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout: f64,
}

impl Default for EmbeddingHttpConfig {
    fn default() -> Self {
        Self {
            endpoint: default_endpoint(),
            key: None,
            max_retries: default_max_retries(),
            timeout: default_timeout(),
        }
    }
}

/// Default embedding endpoint (matches Python `DEFAULT_BASE_URL`).
fn default_endpoint() -> String {
    "http://localhost:8080".to_string()
}

/// Default retry count (matches Python `EmbeddingHttpConfig.max_retries`).
fn default_max_retries() -> u32 {
    3
}

/// Default request timeout in seconds (matches Python).
fn default_timeout() -> f64 {
    60.0
}

/// Provider-specific settings for the OpenAI-compatible backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatibleConfig {
    /// Transport settings; default endpoint is the OpenAI API.
    #[serde(flatten)]
    pub http: EmbeddingHttpConfig,
}

impl Default for OpenAICompatibleConfig {
    fn default() -> Self {
        Self {
            http: EmbeddingHttpConfig {
                endpoint: "https://api.openai.com/v1".to_string(),
                ..EmbeddingHttpConfig::default()
            },
        }
    }
}

/// Provider-specific settings for the TEI native backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeiNativeConfig {
    /// Transport settings.
    #[serde(flatten)]
    pub http: EmbeddingHttpConfig,
    /// Whether to truncate inputs longer than the model context.
    #[serde(default)]
    pub truncate: bool,
    /// Truncation direction: `left` or `right`.
    #[serde(default = "default_truncation_direction")]
    pub truncation_direction: String,
}

impl Default for TeiNativeConfig {
    fn default() -> Self {
        Self {
            http: EmbeddingHttpConfig::default(),
            truncate: false,
            truncation_direction: default_truncation_direction(),
        }
    }
}

/// Default truncation direction (matches Python `TeiNativeConfig`).
fn default_truncation_direction() -> String {
    "right".to_string()
}

/// Provider-specific settings for the Ollama native backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Transport settings; default endpoint is the local Ollama server.
    #[serde(flatten)]
    pub http: EmbeddingHttpConfig,
    /// Whether to truncate inputs longer than the model context.
    #[serde(default = "default_ollama_truncate")]
    pub truncate: bool,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            http: EmbeddingHttpConfig {
                endpoint: "http://localhost:11434".to_string(),
                ..EmbeddingHttpConfig::default()
            },
            truncate: default_ollama_truncate(),
        }
    }
}

/// Default Ollama truncation (matches Python `OllamaConfig`).
fn default_ollama_truncate() -> bool {
    true
}

/// Embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Backend provider; picks which parameter table is read at assembly.
    #[serde(default)]
    pub provider: EmbeddingProvider,
    /// Explicit model id; auto-discovered from the endpoint when omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Native output dimension; required to create an index (vec0 table).
    /// Discovered from the endpoint when omitted — but discovery needs the
    /// embedder, so creating an index without this set is an error until
    /// the embedding layer lands.
    #[serde(default)]
    pub model_dimension: Option<u32>,
    /// Optional output dimension override (payload `dimensions` field).
    #[serde(default)]
    pub output_dimension: Option<u32>,
    /// Batch size for embedding requests.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// OpenAI-compatible backend settings.
    #[serde(default)]
    pub openai_compatible: OpenAICompatibleConfig,
    /// TEI native backend settings.
    #[serde(default)]
    pub tei_native: TeiNativeConfig,
    /// Ollama backend settings.
    #[serde(default)]
    pub ollama: OllamaConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::default(),
            model: None,
            model_dimension: None,
            output_dimension: None,
            batch_size: default_batch_size(),
            openai_compatible: OpenAICompatibleConfig::default(),
            tei_native: TeiNativeConfig::default(),
            ollama: OllamaConfig::default(),
        }
    }
}

/// Default embedding batch size (matches Python `EmbeddingConfig.batch_size`).
fn default_batch_size() -> usize {
    32
}

/// Ranking selector plus all parameter tables.
///
/// Mirrors Python's `RetrievalConfig`: `type` picks which parameter table is
/// read at assembly time; the other tables are always present but ignored
/// ("only loaded when type = ..."), so one config file documents every
/// selectable operator. TOML keys match `lode.toml.example` exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Max results returned by `lode prospect`.
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    /// Per-source score normalization before fusion.
    #[serde(default)]
    pub norm: NormConfig,
    /// How per-source scores combine.
    #[serde(default)]
    pub fusion: FusionConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            norm: NormConfig::default(),
            fusion: FusionConfig::default(),
        }
    }
}

/// Default result cap (matches Python `DEFAULT_TOP_K`).
fn default_top_k() -> u32 {
    10
}

/// Which normalization operator runs before fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NormKind {
    /// Min-max normalize to [0, 1].
    MinMax,
    /// Softmax scaled by temperature.
    #[default]
    Softmax,
}

/// Norm selector plus its parameter tables.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NormConfig {
    /// Which parameter table is read at assembly time.
    #[serde(rename = "type", default)]
    pub kind: NormKind,
    /// Softmax parameters; read when `type = "softmax"`.
    #[serde(default)]
    pub softmax: SoftmaxNormConfig,
}

/// Softmax normalization parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxNormConfig {
    /// Flattens toward uniform as it grows.
    #[serde(default = "default_softmax_temperature")]
    pub temperature: f64,
}

impl Default for SoftmaxNormConfig {
    fn default() -> Self {
        Self {
            temperature: default_softmax_temperature(),
        }
    }
}

/// Default softmax temperature (matches Python `DEFAULT_SOFTMAX_TEMPERATURE`).
fn default_softmax_temperature() -> f64 {
    1.0
}

/// Fusion selector plus its parameter tables.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Which parameter table is read at assembly time.
    #[serde(rename = "type", default)]
    pub kind: FusionKind,
    /// Linear weighted-sum parameters; read when `type = "linear"`.
    #[serde(default)]
    pub linear: LinearFusionConfig,
    /// RRF parameters; read when `type = "rrf"`.
    #[serde(default)]
    pub rrf: RrfFusionConfig,
}

/// Which fusion operator merges the per-source scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionKind {
    /// Weighted sum of per-source scores.
    #[default]
    Linear,
    /// Reciprocal rank fusion.
    Rrf,
}

/// Linear weighted-sum fusion parameters.
///
/// "Factor" rather than "weight": factors may be negative or zero, which
/// weight semantics would not suggest. At most one may be zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearFusionConfig {
    /// Multiplier on the semantic (dense) score.
    #[serde(default = "default_semantic_factor")]
    pub semantic_factor: f64,
    /// Multiplier on the lexical (BM25) score.
    #[serde(default = "default_lexical_factor")]
    pub lexical_factor: f64,
}

impl Default for LinearFusionConfig {
    fn default() -> Self {
        Self {
            semantic_factor: default_semantic_factor(),
            lexical_factor: default_lexical_factor(),
        }
    }
}

/// Default semantic factor (matches Python `DEFAULT_SEMANTIC_FACTOR`).
fn default_semantic_factor() -> f64 {
    0.7
}

/// Default lexical factor (matches Python `DEFAULT_LEXICAL_FACTOR`).
fn default_lexical_factor() -> f64 {
    0.3
}

/// RRF fusion parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RrfFusionConfig {
    /// Rank offset; larger k flattens the reciprocal curve.
    #[serde(default = "default_rrf_k")]
    pub k: u64,
}

impl Default for RrfFusionConfig {
    fn default() -> Self {
        Self { k: default_rrf_k() }
    }
}

/// Default RRF k (matches Python `DEFAULT_RRF_K`).
fn default_rrf_k() -> u64 {
    60
}

/// Assemble the retrieval plan selected by config.
///
/// Mirrors Python's `build_plan`: RRF ranks by position, so normalization is
/// skipped entirely (`norm = None`); min-max and softmax are monotonic, so
/// the rank is identical whether computed on raw or normalized scores. For
/// linear fusion the selected norm is applied per source.
#[must_use]
pub fn build_plan(cfg: &RetrievalConfig) -> crate::index::ranking::RetrievalPlan {
    use crate::index::ranking::{Fusion, Norm, RetrievalPlan};

    if cfg.fusion.kind == FusionKind::Rrf {
        return RetrievalPlan {
            norm: None,
            fusion: Fusion::Rrf {
                k: cfg.fusion.rrf.k,
            },
        };
    }
    let weights = BTreeMap::from([
        (Source::Semantic, cfg.fusion.linear.semantic_factor),
        (Source::Lexical, cfg.fusion.linear.lexical_factor),
    ]);
    let norm = match cfg.norm.kind {
        NormKind::MinMax => Some(Norm::MinMax),
        NormKind::Softmax => Some(Norm::Softmax {
            temperature: cfg.norm.softmax.temperature,
        }),
    };
    RetrievalPlan {
        norm,
        fusion: Fusion::Linear { weights },
    }
}

/// Chunking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Target chunk size in characters.
    #[serde(default = "default_chunk_size")]
    pub size: usize,
    /// Overlap between neighbouring chunks.
    #[serde(default = "default_chunk_overlap")]
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            size: default_chunk_size(),
            overlap: default_chunk_overlap(),
        }
    }
}

/// Default chunk size (matches Python `DEFAULT_CHUNK_SIZE`).
fn default_chunk_size() -> usize {
    1024
}

/// Default chunk overlap (matches Python `DEFAULT_CHUNK_OVERLAP`).
fn default_chunk_overlap() -> usize {
    128
}

/// FTS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsConfig {
    /// FTS5 tokenizer strategy: unicode61, trigram, simple, jieba.
    #[serde(default = "default_tokenizer")]
    pub strategy: String,
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            strategy: default_tokenizer(),
        }
    }
}

/// Default FTS5 tokenizer (matches Python `DEFAULT_TOKENIZER`).
fn default_tokenizer() -> String {
    "simple".to_string()
}

/// Construct the embedding implementation selected by config.
///
/// Mirrors Python's `build_embedder`: construction is side-effect free (no
/// network access); model id and dimension are resolved lazily by the
/// embedder on first use.
pub fn build_embedder(cfg: &EmbeddingConfig) -> crate::Result<Box<dyn Embedder>> {
    match cfg.provider {
        EmbeddingProvider::OpenAICompatible => {
            let client: Box<dyn HttpClient> =
                Box::new(ReqwestHttpClient::new(cfg.openai_compatible.http.timeout)?);
            let embedder =
                OpenAiCompatibleBuilder::new(&cfg.openai_compatible.http.endpoint, client)
                    .model(cfg.model.clone())
                    .dimension(cfg.model_dimension.map(|d| d as usize))
                    .output_dimension(cfg.output_dimension.map(|d| d as usize))
                    .batch_size(cfg.batch_size)
                    .retries(cfg.openai_compatible.http.max_retries)
                    .api_key(cfg.openai_compatible.http.key.clone())
                    .build();
            Ok(Box::new(embedder))
        }
        EmbeddingProvider::TeiNative => {
            let client: Box<dyn HttpClient> =
                Box::new(ReqwestHttpClient::new(cfg.tei_native.http.timeout)?);
            let embedder = TeiNativeBuilder::new(&cfg.tei_native.http.endpoint, client)
                .model(cfg.model.clone())
                .dimension(cfg.model_dimension.map(|d| d as usize))
                .output_dimension(cfg.output_dimension.map(|d| d as usize))
                .batch_size(cfg.batch_size)
                .retries(cfg.tei_native.http.max_retries)
                .api_key(cfg.tei_native.http.key.clone())
                .truncate(cfg.tei_native.truncate)
                .truncation_direction(cfg.tei_native.truncation_direction.clone())
                .build();
            Ok(Box::new(embedder))
        }
        EmbeddingProvider::Ollama => {
            let client: Box<dyn HttpClient> =
                Box::new(ReqwestHttpClient::new(cfg.ollama.http.timeout)?);
            let embedder = OllamaBuilder::new(&cfg.ollama.http.endpoint, client)
                .model(cfg.model.clone())
                .dimension(cfg.model_dimension.map(|d| d as usize))
                .output_dimension(cfg.output_dimension.map(|d| d as usize))
                .batch_size(cfg.batch_size)
                .retries(cfg.ollama.http.max_retries)
                .api_key(cfg.ollama.http.key.clone())
                .truncate(cfg.ollama.truncate)
                .build();
            Ok(Box::new(embedder))
        }
    }
}

/// Ignore rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IgnoreConfig {}
