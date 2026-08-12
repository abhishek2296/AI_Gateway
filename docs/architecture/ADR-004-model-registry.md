# ADR-004: Model Registry Architecture

## Status

Accepted (2026-07-30, updated 2026-08-12)

## Context

The gateway needs a canonical catalog of LLM models (capabilities, limits, enabled
state) that is independent of any single storage backend or provider SDK. Three
parallel metadata types already existed:

- `registry.models.ModelInfo` — gateway catalog value object
- `providers.base.ModelInfo` — vendor `list_models()` DTO
- `models.ai_model.AIModel` — durable ORM row

Routes and chat previously resolved models via `ProviderResolver` and database
defaults without a unified in-memory catalog.

Phase 6.8 hardening consolidated duplicate enums, added metrics/health, optional
routing metadata, and enforced read-only runtime access.

## Decision

1. **Abstract storage contract** — `CatalogModelRegistry` (alias `BaseModelRegistry`)
   defines async `register`, `unregister`, `get`, `list` (with filters), `exists`,
   `providers`, `clear`, and `record_refresh`. `ReadOnlyModelRegistry` exposes
   only read methods + `metrics()` for routes and chat.

2. **In-memory default backend** — `MemoryModelRegistry` with `RLock` is the
   process-local implementation for Phase 6. Redis and database backends are
   deferred but the interface supports them.

3. **Single `ProviderType`** — `src/core/enums.ProviderType` is the only enum
   definition. Registry, schemas, services, and routes import it directly so
   serialization and comparisons never drift between layers.

4. **Single mapping layer** — `registry.mappers` converts provider DTOs to
   registry `ModelInfo`. Database rows overlay enabled/default/display fields
   without redefining capabilities from scratch.

5. **Startup catalog load** — `ModelCatalogLoader` calls each provider's
   `list_models()`, applies DB overlay, seeds `Settings.DEFAULT_*`, then calls
   `record_refresh()` with a timezone-aware UTC timestamp.

6. **Registry-only chat** — `ModelRegistryService.resolve_for_chat()` selects
   and validates models. `AIService` then uses `ProviderResolutionCoordinator`
   only for connection kwargs (API keys, base URLs).

7. **HTTP catalog APIs** — `GET /models`, `GET /models/health`,
   `GET /models/{name}?provider=`, and `GET /providers/{provider}/models`.

8. **Computed metrics** — `RegistryMetrics` derives counts from live registry
   state (`registered_models`, `enabled_models`, `disabled_models`,
   `providers_count`, `default_model`, `last_refresh_time`). No duplicate
   counters that could desync after refresh.

9. **`ModelInfo.supports()`** — Capability checks use `model.supports(capability)`
   (wrapper over `capabilities` frozenset). Filters and routing prep code use
   this helper for readability.

10. **Optional routing metadata (Phase 7 prep)** — `ModelInfo` accepts optional
    `priority`, `cost_per_input_token`, `cost_per_output_token`, `latency_score`,
    `quality_score`, and `tags`. All optional; existing models work without them.
    Phase 7 will consume these fields — Phase 6 does not route on them.

11. **Runtime read-only** — `ModelRegistryService` depends on
    `ReadOnlyModelRegistryView` wrapping the writable backend. Startup and future
    admin refresh use `CatalogModelRegistry` on `app.state.model_registry`.

## Consequences

- Adding Redis/DB backends requires new `CatalogModelRegistry` subclasses only.
- Chat and admin routes depend on startup catalog population; provider/API key
  failures are logged but do not block startup (settings default is seeded).
- Health endpoint reports `healthy`, `degraded` (no refresh timestamp), or
  `unhealthy` (empty or no enabled models).
- Phase 7 routing can read routing metadata and metrics without schema changes.

## Alternatives Considered

- **ORM-only catalog** — Rejected; live provider catalogs would be duplicated.
- **Legacy resolver for chat defaults** — Rejected per product choice; registry
  is the single source of truth for model identity in chat.
- **Duplicate `ProviderType` in registry** — Rejected; unified enum in `core/enums`.
- **Stored metric counters** — Rejected; computed from registry to stay correct
  after refresh.
