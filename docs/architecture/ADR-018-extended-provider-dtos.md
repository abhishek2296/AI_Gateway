# ADR-018: Extended Provider DTOs and Shared HTTP Infrastructure

## Status

Accepted

## Context

Phase 4.6 registered OpenAI, Anthropic, and Gemini as skeleton adapters with minimal DTOs sufficient for text chat. Phase 5 requires full cloud provider implementations supporting vision, tools, structured JSON, streaming deltas, and embeddings — without breaking existing Ollama callers or the `AIService` integration layer.

Cloud adapters also share repetitive concerns: HTTP error mapping, auth headers, retry with backoff, SSE/NDJSON parsing, and model-list caching.

## Decision

1. **Extend DTOs in `providers/base.py`** — Add `TextPart`, `ImagePart`, `ContentPart`, `ToolDefinition`, `ToolCall`, `ToolResult`, and `ResponseFormat`. Extend `ChatMessage`, `ChatRequest`, `ChatResponse`, `ChatStreamChunk`, and `ModelInfo` with optional fields defaulting to backward-compatible values (`content=""`, `parts=None`, etc.).
2. **Shared modules** — Extract reusable infrastructure:
   - `http_errors.py` — `HTTPErrorMapper`
   - `http_auth.py` — Bearer / Anthropic / Gemini header builders
   - `retry.py` — `retry_async()` with exponential backoff on 429/502/503/504
   - `streaming.py` — SSE and NDJSON parsers
   - `validation.py` — `ModelListCache` for `validate_model()`
3. **Refactor Ollama** — Replace inline error mappers with `HTTPErrorMapper`; zero behavior change.
4. **Configuration** — Add retry/cache settings and `ANTHROPIC_API_VERSION` to `core/config.py`.
5. **Capability errors** — Add `UnsupportedCapabilityError` for methods a provider cannot implement (e.g., Anthropic embeddings).
6. **OpenAI API choice** — Implement Chat Completions first (`/v1/chat/completions`); evaluate Responses API in a future ADR.

## Consequences

### Positive

- Cloud providers map vendor payloads into one normalized contract.
- Shared HTTP utilities reduce duplication and keep error semantics consistent.
- DTO defaults preserve compatibility with Phase 4.3 Ollama tests and `AIService`.

### Negative

- Larger DTO surface area to maintain as gateway features expand.
- Extended fields are not yet exposed through HTTP routes/schemas (provider layer only).

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Vendor-specific DTOs per provider | Breaks `BaseProvider` contract and service normalization |
| Vendor SDKs (openai, anthropic) | ADR-014 established httpx-only transport for consistency |
| Separate `capabilities.py` module | `UnsupportedCapabilityError` lives in `exceptions.py` alongside other provider errors |
