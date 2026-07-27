# ADR-019: Cloud Provider Implementations

## Status

Accepted

## Context

Phase 4.6 delivered registered skeleton adapters for OpenAI, Anthropic, and Gemini. Phase 5 completes full httpx-based REST integrations so the gateway can route workloads to cloud vendors using the same `BaseProvider` contract as Ollama.

## Decision

1. **Transport** — httpx only; no vendor SDKs (consistent with ADR-014).
2. **OpenAI** — Chat Completions (`POST /v1/chat/completions`), Embeddings (`POST /v1/embeddings`), model catalog (`GET /v1/models`). SSE streaming via shared `iter_sse_json`. Bearer auth via `Authorization` header.
3. **Anthropic** — Messages API (`POST /v1/messages`). System prompts extracted to top-level `system` field. SSE event dispatch (`content_block_delta`, `message_delta`). Embeddings raise `UnsupportedCapabilityError`. Auth via `x-api-key` + `anthropic-version`.
4. **Gemini** — Generate Content (`POST /v1beta/models/{model}:generateContent`). Model names normalized to `models/{name}`. Auth via `x-goog-api-key`. Safety settings passed through `ChatRequest.provider_options["safetySettings"]`. Streaming supports SSE and NDJSON.
5. **Shared behavior** — All cloud providers use `HTTPErrorMapper`, optional `retry_async`, and `ModelListCache` for `list_models` / `validate_model`.
6. **Health checks** — Lightweight model-list probe; auth/transport failures return `healthy=False` without raising.
7. **Scope boundary** — No route, schema, repository, or migration changes in Phase 5.
8. **Roadmap renumbering** — Phase 5 = Cloud Providers; Model Registry moves to Phase 6.

## Consequences

### Positive

- Four registered providers (`ollama`, `openai`, `anthropic`, `gemini`) with full adapter coverage.
- respx unit tests (115 total) validate normalization without live API keys.
- `ProviderConfigResolver` builds Anthropic/Gemini kwargs including API version.

### Negative

- Token estimation remains unsupported for cloud providers without adding tiktoken or vendor count APIs.
- Anthropic extended thinking is stored in `ChatResponse.details` only — not exposed at HTTP layer yet.
- Gemini tool-result mapping uses simplified function-response shape.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| OpenAI Responses API | Higher migration cost; Chat Completions sufficient for gateway parity |
| Anthropic SDK | Violates httpx-only decision |
| Gemini query-string API key | Header auth is current Google REST default; documented in provider guides |
