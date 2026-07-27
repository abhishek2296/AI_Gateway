# ADR-014: Ollama Provider Adapter

## Status

Accepted

## Context

Phase 4.2 delivered `ProviderRegistry` and `ProviderFactory`. Phase 4.3 needs the first concrete adapter so subsequent service wiring can resolve `"ollama"` from the registry. The legacy `OllamaService` uses the `ollama` Python SDK; the new adapter must use raw HTTP via `httpx` to stay consistent with future providers and keep transport logic isolated.

## Decision

1. **Module** — `backend/src/providers/ollama.py` implements `OllamaProvider(BaseProvider)`.
2. **HTTP client** — `httpx.AsyncClient` with configurable `base_url` and `timeout`. Callers may inject a shared client; the provider tracks ownership and only closes internally created clients.
3. **Endpoints** — Ollama REST API:
   - `GET /api/tags` — health and model catalog
   - `POST /api/chat` — chat (non-streaming and streaming via `stream` flag)
   - `POST /api/embeddings` — embeddings
4. **DTO mapping** — All responses mapped to frozen provider DTOs (`ChatResponse`, `ChatStreamChunk`, `EmbeddingsResponse`, `ModelInfo`, `HealthCheckResult`, `TokenUsage`). Raw Ollama JSON never leaves the adapter.
5. **Streaming** — `client.stream()` + `response.aiter_lines()`; parse NDJSON incrementally and yield `ChatStreamChunk` without buffering the full response.
6. **Error translation** — Map `httpx` failures to `ProviderError` subclasses; never leak vendor exceptions.
7. **Self-registration** — `register_provider(OllamaProvider)` at module import; `providers/__init__.py` imports the module to trigger registration.
8. **Scope boundary** — No changes to services, routes, DI, or repositories in this phase.

## Consequences

### Positive

- First end-to-end provider adapter validates the Phase 4.1 contract and Phase 4.2 registry.
- HTTP-only transport aligns with OpenAI/Anthropic adapter patterns to follow.
- respx-based unit tests run without a live Ollama server.

### Negative

- Temporary overlap with legacy `OllamaService` until service layer migration.
- Ollama `/api/embeddings` vs `/api/embed` naming must match deployed Ollama version (REST docs use `/api/embed`; requirement specifies `/api/embeddings`).

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Reuse `ollama` Python SDK | Couples adapter to SDK lifecycle; harder to map errors uniformly |
| Implement in `services/` | Violates provider layer boundary (ADR-003) |
| Singleton provider instance | Conflicts with factory pattern; shared mutable HTTP state across requests |
