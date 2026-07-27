# ADR-017: Multi-Vendor Provider Skeletons

## Status

Accepted

## Context

Phase 4.1–4.5 established provider abstraction, Ollama implementation, service integration, and database-backed resolution. Before implementing full OpenAI, Anthropic, and Gemini HTTP integrations, the gateway needs registered adapter classes so the registry, factory, resolver, and tests can exercise multi-vendor paths.

## Decision

1. **Skeleton adapters** — `OpenAIProvider`, `AnthropicProvider`, and `GeminiProvider` implement the full `BaseProvider` contract but raise `NotImplementedError` for unimplemented API methods with explicit messages.
2. **Streaming** — `stream_chat()` raises `StreamingNotSupportedError` until vendor streaming is implemented.
3. **Health** — `health_check()` returns `HealthCheckResult(healthy=False, message=...)` rather than raising, so probes report status without crashing callers.
4. **Auto-registration** — Each module calls `register_provider()` at import time.
5. **`HTTPProviderMixin`** — Shared `httpx.AsyncClient` ownership, `close()`, and async context manager; `OllamaProvider` refactored to inherit without behavior change.
6. **Constructor validation** — `require_non_empty_string()` helper for required credentials; provider-specific kwargs only (no unnecessary arguments).
7. **Independence** — No provider imports another provider module.

## Consequences

### Positive

- Registry lists all planned vendors; factory and resolver can target them immediately.
- Future implementation is scoped to individual provider files.
- Shared HTTP lifecycle reduces duplication and drift.

### Negative

- Selecting a skeleton provider at runtime yields `NotImplementedError` on chat — acceptable until 4.7.
- `ProviderType` enum still limited to `ollama` for API responses until extended.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Stub providers that return empty responses | Hides missing implementation; violates explicit failure requirement |
| Single generic skeleton class | Loses provider-specific constructor contracts and registration keys |
| Implement all vendors in one phase | Too large; delays registry/factory validation |
