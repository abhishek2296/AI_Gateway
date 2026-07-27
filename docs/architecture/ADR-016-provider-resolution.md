# ADR-016: Database-Backed Provider Resolution

## Status

Accepted

## Context

Phase 4.4 wired `AIService` to `ProviderFactory` using settings-only provider and model selection. Phase 4.5 introduces database-backed resolution using existing Phase 3 repositories (`Provider`, `AIModel`, `APIKey`, `ProviderConfiguration`) without schema changes or API modifications.

## Decision

1. **`ProviderResolver`** — Framework-agnostic class that resolves provider name, model name, and credential references using an ordered strategy chain:
   - Request override → database default → settings → hardcoded `"ollama"`.
2. **`ProviderConfigResolver`** — Translates resolved ORM rows into `ProviderFactory.create()` kwargs (e.g. `base_url`, `timeout`, `api_key` from env var names). Keeps provider-specific parameter knowledge out of `AIService`.
3. **`ProviderResolutionCoordinator`** — Combines resolver, config resolver, and TTL cache; opens a short-lived DB session on cache miss.
4. **`ProviderResolutionCache`** — In-memory TTL cache (default 60s) for resolved **configuration snapshots** only.
5. **`ProviderRepository.get_default_provider()`** — Returns the active provider owning a default model, or the first active provider by name.
6. **`AIService`** — Delegates selection to the coordinator; no direct repository access or provider-specific imports.

## Consequences

### Positive

- Admin can change default provider/model in PostgreSQL without redeploying settings.
- Cache reduces repository load under steady traffic.
- Resolver and config resolver are reusable outside FastAPI (CLI, workers).

### Negative

- Settings and database can disagree; precedence rules must be documented and tested.
- `ProviderType` enum still limits API-exposed provider names until extended.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| AIService queries repositories directly | Violates separation; harder to test and reuse |
| Cache provider adapter instances | Shared mutable HTTP state; conflicts with factory design |
| Redis cache | Unnecessary for current scale; in-memory TTL sufficient |
