# AGENTS.md

## Project context

This directory contains the Rust rewrite of the project.

The goal is to preserve the original semantics and user-facing behavior,
not to translate the Python implementation line by line.

When the existing behavior and the new design differ:
- Analyze the semantic differences
- Determine whether the difference is intentional or accidental
- When appropriate, ask for clarification before making semantic changes.

Use Rust-native designs where appropriate:
- Prefer idiomatic Rust ownership and error handling
- Replace Python-specific patterns with suitable Rust abstractions
- Improve unclear or accidental designs when there is a clear benefit

## Code quality

### Modernization

The following are project preferences for writing modern Rust.
They are guidelines, not absolute rules. Use judgement when an exception improves clarity, performance, or API design.

#### Ownership

##### Prefer owned data for long-lived objects

Long-lived structs should usually own their data instead of storing references.

**Recommended**

```rust
struct Provider {
    base_url: String,
    model: String,
}
```

**Be careful**

```rust
struct Provider<'a> {
    base_url: &'a str,
    model: &'a str,
}
```

**Reason**

- Reduce lifetime propagation through the codebase
- Make APIs easier to compose

References are still preferred for short-lived operations.

**Recommended**

```rust
fn parse(input: &str) {
}
```

---

#### Configuration

##### Configuration objects should own their data

Configuration is usually created, stored, and passed between layers.
Prefer owned values.

**Recommended**

```rust
struct EmbeddingConfig {
    model: Option<String>,
    endpoint: String,
}
```

**Be careful**

```rust
struct EmbeddingConfig<'a> {
    model: Option<&'a str>,
    endpoint: &'a str,
}
```

---

#### Error handling

##### Prefer domain errors over infrastructure errors

Core layers should expose meaningful domain errors instead of leaking implementation details.

**Avoid**

```rust
Result<Response, reqwest::Error>
```

**Prefer**

```rust
Result<Response, EmbeddingError>
```

**Reason**

- Keep provider implementation details isolated
- Allow upper layers to handle meaningful failures
- Make errors stable when implementations change

External errors should usually be wrapped:

```rust
EmbeddingError::RequestFailed {
    #[source]
    source: reqwest::Error,
}
```

---

#### Strings

##### Use `&str` for borrowing and `String` for ownership

Use borrowed strings for temporary operations.

**Recommended**

```rust
fn normalize(text: &str) -> String {
}
```

Use owned strings when storing data.

**Recommended**

```rust
struct Document {
    title: String,
}
```

Avoid storing references unless the lifetime relationship is simple and intentional.

---

#### Paths

##### Prefer filesystem types over raw strings

Filesystem paths should use Rust path types.

**Avoid**

```rust
struct FileInfo {
    path: String,
}
```

**Prefer**

```rust
struct FileInfo {
    path: PathBuf,
}
```

**Reason**

- Preserve platform-specific behavior
- Avoid manual path manipulation
- Make filesystem intent explicit

---

#### External APIs

##### Prefer typed wrappers for meaningful values

Avoid using plain strings for values with different meanings.

**Avoid**

```rust
fn connect(model: String, url: String) {
}
```

A model identifier and a URL are both strings, but they represent different concepts.

Prefer typed wrappers when ambiguity is possible.

**Example**

```rust
struct ModelId(String);

struct BaseUrl(Url);
```

---

#### Builders

##### Builders should usually own configuration

Builders often outlive the call that creates them.

**Recommended**

```rust
struct ProviderBuilder {
    model: Option<String>,
}
```

**Be careful**

```rust
struct ProviderBuilder<'a> {
    model: Option<&'a str>,
}
```

Prefer ownership unless borrowing provides a clear benefit.

---

#### Traits

##### Keep traits focused

Avoid large traits that combine unrelated responsibilities.

**Avoid**

```rust
trait Provider {
    fn request(&self);
    fn parse(&self);
    fn cache(&self);
    fn retry(&self);
}
```

Prefer smaller traits with clear responsibilities.

**Example**

```rust
trait Embedder {
}

trait HealthCheck {
}
```

---

#### Async

##### Avoid unnecessary borrowing across async boundaries

Async code often requires data to live longer than a single scope.

Prefer:

- Owned state
- `Arc`
- Cloneable configuration objects

Be careful with references captured by async tasks.

---

#### Clone

##### Clone intentionally

Do not add `.clone()` only to satisfy the borrow checker without understanding the ownership model.
Do not optimize clones prematurely.
Prefer clear ownership over complex borrowing unless profiling shows a problem.

A clone is acceptable when:

- The data is small
- Ownership transfer would complicate the design
- Lifetime complexity is not worth avoiding the allocation

Document non-obvious clones when necessary.

---

#### Abstraction

##### Avoid premature abstraction

Do not introduce traits, wrappers, or additional layers without a clear need.

Prefer simple concrete implementations when:
- There is only one implementation
- The abstraction does not improve testability or separation
- The additional complexity outweighs the benefit

Introduce abstractions when they represent a real boundary:
- Multiple implementations exist
- External behavior needs isolation
- Testing requires substitution

---

#### Lint expectations

Prefer explicit lint expectations over silent suppression.

**Preferred**

```rust
#[expect(
    clippy::module_name_repetitions,
    reason = "Public API names intentionally follow domain terminology"
)]
```

Avoid unexplained:

```rust
#[allow(clippy::some_rule)]
```

Exceptions are acceptable, but they should explain why the rule does not apply.

---

#### API design

Prefer APIs that make invalid states harder to represent.

Consider:

- Typed wrappers instead of ambiguous primitives
- Domain-specific errors
- Explicit ownership boundaries
- Clear separation between internal implementation and external interfaces


### Checks

#### Common

```bash
cargo check
cargo fmt
cargo test
```

#### Before commit

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

#### Rust idiom review

New modules and substantial refactors should enable:
```rust
#![warn(clippy::pedantic)]
```

Each pedantic warning should be reviewed:

- Fix the issue when it indicates a correctness, maintainability, or idiomatic Rust concern.
- Otherwise, suppress it explicitly with:

```rust
#[expect(clippy::some_rule, reason = "...")]
```

Avoid unexplained lint suppression.
