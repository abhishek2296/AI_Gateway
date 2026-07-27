# ADR-015: Provider Service Integration

## Status

Accepted

## Context

Phase 4.1–4.3 established `BaseProvider`, registry/factory, and `OllamaProvider`. HTTP routes still depended on `OllamaService` and `BaseLLMService`, bypassing the provider layer. Phase 4.4 migrates the application service layer to resolve and invoke providers through `ProviderFactory` without changing public API contracts.

## Decision

1. **`AIService`** — New provider-agnostic service in `backend/src/services/ai_service.py`.
   - Resolves provider via `settings.DEFAULT_PROVIDER` (caller override supported for future use).
   - Resolves model via `settings.DEFAULT_MODEL` when not specified.
   - Creates providers through injected `ProviderFactory`.
   - Maps provider DTOs to legacy dict payloads consumed by existing routes.
   - Maps `ProviderError` to existing `OllamaConnectionException` / `LLMResponseException`.
2. **`ChatService`** — Thin delegate to `AIService`; route signature unchanged.
3. **`LLMHealthAdapter`** — Implements legacy `BaseLLMService` for `/health` without route edits.
4. **Dependency injection** — Cached `ProviderFactory` and `AIService` in `api/dependencies.py`.
5. **Lifecycle** — `_execute_provider_call` uses `async with provider` when supported; otherwise calls `close()` in `finally`.
6. **No database resolution** — Provider/model selection from settings only; DB lookup deferred to Phase 5+.
7. **Legacy `OllamaService`** — Retained in codebase but no longer wired to routes or lifespan.

## Consequences

### Positive

- Routes and Pydantic schemas unchanged; clients see identical behavior.
- New providers require registry registration + settings kwargs mapping, not `AIService` rewrites.
- Centralized logging, timing, error translation, and cleanup.

### Negative

- Temporary `_provider_kwargs` string mapping in `AIService` until a config registry exists.
- `ProviderType` enum must be extended when new providers are exposed via API.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Replace routes to use `AIService` directly | Violates no-API-change constraint |
| Keep `OllamaService` in DI | Does not meet Phase 4 factory integration goal |
| Singleton provider instances in `AIService` | Conflicts with factory fresh-instance pattern |
