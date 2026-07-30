# ADR-004: Model Registry Architecture

## Status

Accepted (2026-07-30)

## Context

The gateway needs a canonical catalog of LLM models (capabilities, limits, enabled
state) that is independent of any single storage backend or provider SDK. Three
parallel metadata types already existed:

- `registry.models.ModelInfo` — gateway catalog value object
- `providers.base.ModelInfo` — vendor `list_models()` DTO
- `models.ai_model.AIModel` — durable ORM row

Routes and chat previously resolved models via `ProviderResolver` and database
defaults without a unified in-memory catalog.

## Decision

1. **Abstract storage contract** — `BaseModelRegistry` (async) defines
   `register`, `unregister`, `get`, `list` (with filters), `exists`, `providers`,
   `clear`.

2. **In-memory default backend** — `MemoryModelRegistry` with `RLock` is the
   process-local implementation for Phase 6.3–6.8. Redis and database backends
   are deferred but the interface supports them.

3. **Single mapping layer** — `registry.mappers` converts provider DTOs to
   registry `ModelInfo`. Database rows overlay enabled/default/display fields
   without redefining capabilities from scratch.

4. **Startup catalog load** — `ModelCatalogLoader` calls each provider's
   `list_models()`, applies DB overlay, and seeds `Settings.DEFAULT_*` so the
   catalog is never empty.

5. **Registry-only chat** — `ModelRegistryService.resolve_for_chat()` selects
   and validates models. `AIService` then uses `ProviderResolutionCoordinator`
   only for connection kwargs (API keys, base URLs).

6. **HTTP catalog APIs** — `GET /models`, `GET /models/{name}?provider=`, and
   `GET /providers/{provider}/models` expose the registry with pagination-ready
   responses.

## Consequences

- Adding Redis/DB backends requires new `BaseModelRegistry` subclasses only.
- Chat and admin routes depend on startup catalog population; provider/API key
  failures are logged but do not block startup (settings default is seeded).
- Three `ProviderType` enums remain (`registry`, `core`, string keys); map via
  `.value` at boundaries.

## Alternatives Considered

- **ORM-only catalog** — Rejected; live provider catalogs would be duplicated.
- **Legacy resolver for chat defaults** — Rejected per product choice; registry
  is the single source of truth for model identity in chat.
