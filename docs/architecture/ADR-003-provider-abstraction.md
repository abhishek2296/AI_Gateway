# ADR-003: Provider Abstraction Layer

## Status

Accepted

## Context

Phase 4 introduces multi-provider support. The gateway currently uses `BaseLLMService` in `services/` with a minimal Ollama implementation tied to HTTP routes. A vendor-neutral provider layer is needed before adding OpenAI, Anthropic, Gemini, Azure OpenAI, and Bedrock.

## Decision

1. **New package** — `backend/src/providers/` separate from `services/` and `schemas/`.
2. **Abstract contract** — `BaseProvider` defines async methods: `chat`, `stream_chat`, `embeddings`, `list_models`, `health_check`.
3. **Normalized DTOs** — frozen dataclasses (`ChatRequest`, `ChatResponse`, etc.) in `providers/base.py` decouple vendor APIs from HTTP Pydantic schemas.
4. **Exception hierarchy** — `ProviderError` and subclasses in `providers/exceptions.py`; services map these to HTTP responses.
5. **No implementation in this phase** — concrete adapters and registry come in subsequent Phase 4 sub-phases. Existing `OllamaService` / routes remain unchanged until wired.

## Consequences

### Positive

- One interface for all LLM backends.
- Gateway services can depend on `BaseProvider` without vendor imports.
- Errors are classified consistently for routing, retries, and observability.

### Negative

- Temporary overlap with legacy `BaseLLMService` until migration completes.
- Dataclass DTOs parallel Pydantic schemas; mapping belongs in the service layer.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Extend `BaseLLMService` only | Too narrow (no embeddings/streaming/models catalog) |
| Pydantic models in provider layer | Couples adapters to HTTP validation concerns |
| Single mega `call()` method | Poor type safety and harder provider-specific mapping |
