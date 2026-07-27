# AI Gateway — Project Memory

Living document tracking objectives, decisions, and progress. Append new entries; never delete history.

---

## Project Objective

Build a **provider-agnostic AI Gateway** — a production-grade HTTP API that normalizes access to multiple LLM providers (Ollama, OpenAI, Anthropic, Gemini, Azure OpenAI, AWS Bedrock) for chat, embeddings, vision, RAG, agents, and future AI workloads.

**This is NOT a chatbot or coding assistant.**

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14+ |
| Package manager | uv |
| Web framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Settings | pydantic-settings |
| ORM | SQLAlchemy 2.x (async) |
| DB driver | asyncpg |
| Database | PostgreSQL 17 |
| LLM (current) | Ollama (`qwen3:8b`) |
| Containerization | Docker Compose (Postgres only) |
| Migrations | Alembic (configured) |

---

## Folder Structure (models)

```
backend/
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 20260723_2108_initial_schema.py
│       ├── 20260723_2200_gateway_operational_models.py
│       └── 20260723_2345_phase3_hardening_constraints.py
└── src/models/
    ├── base.py
    ...
└── src/repositories/
    ├── base.py
    ...
└── src/unit_of_work/
    ├── base.py
    ...
└── tests/
    ├── conftest.py
    └── integration/
```

**10 ORM entities** + **11 repository classes** + **Unit of Work** + **41 integration tests**.

---

## Completed Phases

| Phase | Summary |
|-------|---------|
| 1 — Project Foundation | FastAPI, config, logging, middleware, Docker, env |
| 2 — AI Gateway Core | Ollama provider, chat/health, exceptions, lifespan |
| 3.1 — Database Infrastructure | Async engine, session factory, `get_session()` |
| 3.2.1 — ORM Foundation | `Base`, `TimestampMixin` |
| 3.3 — Core Domain Models | `Provider`, `AIModel` + relationship |
| 3.4 — Provider & Model Configuration | Refactors, `ProviderConfiguration`, `AIModelConfiguration` |
| 3.5 — Conversation Domain Models | `ChatSession`, `Message`, `PromptTemplate` |
| 3.6 — Alembic | Async env, initial migration `a3f6c2d18e01` (7 tables) |
| 3.7 — Gateway Operational Models | `APIKey`, `UsageRecord`, `ProviderHealth` + migration `b7e4d9f21c03` |
| 3.8 — Repository Pattern | `BaseRepository` + 10 entity repositories |
| 3.9 — Unit of Work | `BaseUnitOfWork`, `AsyncUnitOfWork` |
| 3.10 — Persistence Layer Testing | pytest suite, 41 integration tests |
| 3.11 — Persistence Hardening | Unique `request_id`, one default model/key per provider |

---

## Current Work

**Phase 5 — Cloud Provider Implementations** (complete)

**Next:** Phase 6 — Model Registry

---

## Memory Log

### 2026-07-27 — Phase 5 Cloud Provider Implementations

**Phase:** 5.0–5.7

**Objective:** Full httpx REST adapters for OpenAI, Anthropic, and Gemini with shared HTTP infrastructure, extended DTOs, contract tests, and documentation.

**Files created:**
- `backend/src/providers/http_errors.py`, `http_auth.py`, `retry.py`, `streaming.py`, `validation.py`
- Provider tests and fixtures under `backend/tests/unit/providers/`
- `docs/architecture/ADR-018-extended-provider-dtos.md`, `ADR-019-cloud-provider-implementations.md`
- `docs/providers/openai.md`, `anthropic.md`, `gemini.md`

**Files modified:**
- `backend/src/providers/base.py`, `exceptions.py`, `ollama.py`, `openai.py`, `anthropic.py`, `gemini.py`, `__init__.py`
- `backend/src/services/provider_config_resolver.py`
- `backend/src/core/config.py`, `core/enums.py`
- `docs/ARCHITECTURE.md`, `ROADMAP.md`, `CHANGELOG.md`, `.env.example`

**Decisions:**
- httpx-only transport (no vendor SDKs); OpenAI Chat Completions API first.
- Shared `HTTPErrorMapper`, retry, streaming parsers, model-list cache.
- Anthropic embeddings raise `UnsupportedCapabilityError`.
- Gemini safety settings via `ChatRequest.provider_options`.
- ROADMAP renumbered: Phase 5 = cloud providers; Model Registry → Phase 6.

**Verification:** 115 unit tests passed.

**Next task:** Phase 6 — Model Registry.

### 2026-07-27 — Phase 4.6 Provider Skeletons

**Phase:** 4.6

**Objective:** Register OpenAI, Anthropic, and Gemini as production-ready skeleton adapters. Only Ollama remains fully functional.

**Files created:**
- `backend/src/providers/openai.py`, `anthropic.py`, `gemini.py`, `http_mixin.py`
- `backend/tests/unit/providers/test_provider_skeletons.py`
- `docs/architecture/ADR-017-provider-skeletons.md`

**Files modified:**
- `backend/src/providers/ollama.py`, `__init__.py`
- `backend/src/services/ai_service.py`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:**
- Skeletons register via `register_provider()` at import; `NotImplementedError` for unimplemented methods.
- `health_check()` returns `healthy=False` instead of raising — graceful degradation for probes.
- `HTTPProviderMixin` shared across HTTP providers; Ollama refactored without behavior change.
- Providers remain independent — no cross-imports between vendor modules.

**Verification:** 76 unit tests passed (29 skeleton + 47 existing).

**Next task:** Implement OpenAI / Anthropic / Gemini API calls (4.7).

### 2026-07-27 — Phase 4.5 Database-Backed Provider Resolution

**Phase:** 4.5

**Objective:** Replace settings-only provider/model selection with database-backed resolution via existing repositories. No API, schema, route, or provider implementation changes.

**Files created:**
- `backend/src/services/provider_resolver.py`, `provider_config_resolver.py`, `provider_resolution_coordinator.py`, `resolution_types.py`
- `backend/src/core/provider_resolution_cache.py`
- `backend/tests/unit/services/test_provider_resolution.py`
- `docs/architecture/ADR-016-provider-resolution.md`

**Files modified:**
- `backend/src/services/ai_service.py`
- `backend/src/repositories/provider_repository.py`
- `backend/src/core/config.py`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:**
- Ordered strategy chain for provider precedence (request → DB → settings → hardcoded).
- `ProviderConfigResolver` owns provider-specific constructor kwargs; `AIService` stays agnostic.
- TTL cache stores configuration snapshots only — never provider instances.
- `ProviderResolver` is FastAPI-independent for CLI/worker reuse.

**Verification:** 47 unit tests passed.

**Next task:** Additional provider adapters (4.6) or Phase 5 model registry enhancements.

### 2026-07-27 — Phase 4.4 Provider Service Integration

**Phase:** 4.4

**Objective:** Route AI operations through `ProviderFactory` / `BaseProvider` via `AIService`. No API, schema, repository, or route changes.

**Files created:**
- `backend/src/services/ai_service.py`, `llm_adapter.py`
- `backend/tests/unit/services/test_ai_service.py`
- `docs/architecture/ADR-015-provider-service-integration.md`

**Files modified:**
- `backend/src/services/chat_service.py`
- `backend/src/api/dependencies.py`
- `backend/src/core/config.py`, `backend/src/core/lifespan.py`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:**
- `AIService` resolves provider/model from settings (`DEFAULT_PROVIDER`, `DEFAULT_MODEL`); DB lookup deferred.
- `_execute_provider_call` centralizes logging, timing, error mapping, and provider cleanup.
- `LLMHealthAdapter` preserves `/health` route contract without route edits.
- Legacy `OllamaService` retained but unwired.

**Verification:** 35 unit tests passed (9 AIService + 26 provider).

**Next task:** Database-backed provider/model resolution (Phase 5) or additional provider adapters (4.5).

### 2026-07-27 — Phase 4.3 Ollama Provider Adapter

**Phase:** 4.3

**Objective:** First concrete `BaseProvider` implementation using Ollama REST via httpx. No service/route/DI changes.

**Files created:**
- `backend/src/providers/ollama.py`
- `backend/tests/unit/providers/test_ollama_provider.py`
- `docs/architecture/ADR-014-ollama-provider.md`

**Files modified:**
- `backend/src/providers/__init__.py`, `backend/pyproject.toml`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:**
- httpx over `ollama` SDK for uniform transport across future adapters.
- Provider owns HTTP lifecycle only; business logic stays in services.
- `register_provider(OllamaProvider)` at module import for self-registration.
- ADR numbered 014 (ADR-004 slot reserved for Phase 5 model registry).

**Verification:** 26 provider unit tests passed (14 Ollama + 12 registry).

**Next task:** Wire `OllamaProvider` into services/DI, or add OpenAI/Anthropic adapters (4.4).

### 2026-07-24 — Phase 4.2 Provider Registry & Factory

**Phase:** 4.2

**Objective:** Thread-safe registry of provider classes and a small factory for on-demand instantiation. No service/route changes, no Ollama HTTP.

**Files created:**
- `backend/src/providers/registry.py`, `factory.py`
- `backend/tests/unit/providers/mock_provider.py`, `test_provider_registry.py`

**Files modified:**
- `backend/src/providers/__init__.py`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:**
- Registry stores **classes**, not instances — shared catalog, no shared mutable state across requests.
- Factory does **not** cache instances — each call gets a clean lifecycle and request-scoped constructor kwargs.
- `register_provider()` on the default registry prepares import-time self-registration by future provider modules.

**Verification:** 12 unit tests passed (`tests/unit/providers/test_provider_registry.py`).

**Next task:** `OllamaProvider` implementing `BaseProvider` (4.3).

### 2026-07-24 — Phase 4.1 Provider Abstraction

**Phase:** 4.1

**Objective:** Vendor-neutral provider contract and exception hierarchy. No implementations, no route/service changes.

**Files created:**
- `backend/src/providers/base.py`, `exceptions.py`, `__init__.py`
- `docs/architecture/ADR-003-provider-abstraction.md` (populated)

**Files modified:**
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:** See [ADR-003](architecture/ADR-003-provider-abstraction.md).

**Next task:** Ollama adapter implementing `BaseProvider` (4.3).

### 2026-07-23 — Phase 3.11 Persistence Hardening

**Phase:** 3.11

**Objective:** Final Phase 3 schema hardening before Phase 4.

**Files created:**
- `backend/alembic/versions/20260723_2345_phase3_hardening_constraints.py` (revision `c8f5e2a31d04`)

**Files modified:**
- `backend/src/models/usage_record.py`, `ai_model.py`, `api_key.py`
- `backend/tests/conftest.py`, `backend/tests/integration/test_orm_models.py`
- `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `CHANGELOG.md`, `PROJECT_MEMORY.md`

**Constraints added:**
- `uq_usage_records_request_id` — prevents duplicate billing rows
- `uq_ai_models_one_default_per_provider` — partial unique on `(provider_id) WHERE is_default`
- `uq_api_keys_one_default_per_provider` — partial unique on `(provider_id) WHERE is_default`

**Verification:** Alembic upgrade/downgrade; full pytest suite (41 passed).

**Next task:** Phase 4 — Multi-Provider Architecture.

### 2026-07-23 — Phase 3.10 Persistence Layer Testing

**Phase:** 3.10

**Objective:** Comprehensive pytest integration suite for persistence layer. No service or API tests.

**Files created:**
- `backend/tests/conftest.py`, `factories.py`, `helpers.py`, `README.md`
- `backend/tests/integration/test_alembic.py`, `test_orm_models.py`, `test_repositories.py`, `test_unit_of_work.py`

**Files modified:**
- `backend/pyproject.toml`, `backend/src/core/config.py`, `.env.example`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`
- `.cursor/rules/07-testing.mdc`

**Decisions:**
- PostgreSQL only (no SQLite); separate `TEST_DATABASE_URL`
- Session-scoped Alembic migrate via subprocess; per-test transaction rollback
- `committed_session` + truncate for UoW commit tests
- Session-scoped asyncio event loop for engine/fixture compatibility

**Verification:** `uv run pytest` — 38 passed.

**Next task:** Phase 4 — Multi-Provider Architecture.

### 2026-07-23 — Phase 3.9 Unit of Work

**Phase:** 3.9

**Objective:** Implement Unit of Work to coordinate repositories and transaction boundaries. No services, DI, or business logic.

**Files created:**
- `backend/src/unit_of_work/base.py`, `unit_of_work.py`, `__init__.py`
- `docs/architecture/ADR-013-unit-of-work-pattern.md`

**Files modified:**
- `backend/src/repositories/base.py` (docstring)
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`
- `.cursor/rules/04-database.mdc`

**Decisions:** See [ADR-013](architecture/ADR-013-unit-of-work-pattern.md).
- `BaseUnitOfWork` ABC; `AsyncUnitOfWork` registers all 10 repos on one session
- Explicit commit only; rollback on exception in context manager
- `close_session=True` by default; configurable for outer session owners

**Verification:** Imports succeed; repository properties cached on UoW instance.

**Next task:** Testing (3.10).

### 2026-07-23 — Phase 3.8 Repository Pattern

**Phase:** 3.8

**Objective:** Implement async repository layer for all 10 ORM entities. No services, DI, or Unit of Work.

**Files created:**
- `backend/src/repositories/base.py` + 10 entity repository modules + `__init__.py`
- `docs/architecture/ADR-012-repository-pattern.md`

**Files modified:**
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`
- `.cursor/rules/04-database.mdc`

**Decisions:** See [ADR-012](architecture/ADR-012-repository-pattern.md).
- Generic `BaseRepository[ModelT]` for shared CRUD
- SQLAlchemy 2.x `select()` only; flush allowed, no commit/rollback in repos
- Session injected via constructor; FastAPI DI deferred
- Entity repos contain persistence queries only

**Verification:** All repository imports succeed.

**Next task:** Unit of Work (3.9).

### 2026-07-23 — Phase 3.7 Gateway Operational Models

**Phase:** 3.7

**Objective:** Complete remaining operational ORM entities (`APIKey`, `UsageRecord`, `ProviderHealth`). No repositories, services, or APIs.

**Files created:**
- `backend/src/models/api_key.py`, `usage_record.py`, `provider_health.py`
- `backend/alembic/versions/20260723_2200_gateway_operational_models.py` (revision `b7e4d9f21c03`)
- `docs/architecture/ADR-011-gateway-operational-models.md`

**Files modified:**
- `backend/src/models/provider.py`, `ai_model.py`, `chat_session.py`, `__init__.py`
- `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md`, `CHANGELOG.md`, `ROADMAP.md`

**Decisions:** See [ADR-011](architecture/ADR-011-gateway-operational-models.md).
- `APIKey.api_key_env` stores env var name only; unique `(provider_id, name)`
- `UsageRecord`: session FK `SET NULL`; provider/model FK `RESTRICT`; `Numeric(12,6)` for cost
- `ProviderHealth`: historical snapshots (1:N), latest by `checked_at`
- All relationships `lazy="selectin"`

**Verification:** 10 tables on `Base.metadata`; offline Alembic upgrade SQL verified.

**Next task:** Repository pattern (3.8).

### 2026-07-23 — Phase 3.6 Alembic Migration Infrastructure

**Phase:** 3.6

**Objective:** Configure Alembic with async env and initial schema migration for all 7 ORM tables.

**Files created:**
- `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/script.py.mako`, `backend/alembic/README`
- `backend/alembic/versions/20260723_2108_initial_schema.py` (revision `a3f6c2d18e01`)
- `docs/architecture/ADR-010-alembic-async-migrations.md`

**Decisions:** See [ADR-010](architecture/ADR-010-alembic-async-migrations.md).
- Async Alembic env with asyncpg (no psycopg2)
- Metadata via `import src.models`; URL from `settings.DATABASE_URL`
- Initial migration reviewed against ORM metadata; upgrade/downgrade SQL verified offline

**Verification:** `upgrade head --sql` and `downgrade a3f6c2d18e01:base --sql` confirmed 7 tables + correct drop order. Live `upgrade` requires `.env` with credentials matching Docker Postgres.

**Next task:** APIKey entity or repository pattern (3.7).

### 2026-07-23 — Phase 3.5 Conversation Domain Models

**Phase:** 3.5

**Objective:** Implement conversation ORM entities. No repositories, APIs, or migrations.

**Files changed:**
- Created: `chat_session.py`, `message.py`, `prompt_template.py`
- Updated: `provider.py`, `ai_model.py`, `models/__init__.py`
- Docs: ARCHITECTURE, PROJECT_MEMORY, CHANGELOG, ROADMAP, ADR-009

**Decisions:** See [ADR-009](architecture/ADR-009-conversation-domain-models.md).
- `session_uuid` as UUID; denormalized `provider_id` on sessions
- `extra_metadata` → DB column `metadata`
- Message `role` as string; order by `created_at`
- PromptTemplate unique `(name, version)`; standalone catalog

**Next task:** APIKey entity or Alembic initial migration.

### 2026-07-23 — Phase 3.4 Provider & Model Configuration

**Phase:** 3.4

**Objective:** Refactor Phase 3.3 models and add 1:1 configuration entities. ORM only.

**Files changed:**
- Created: `ai_model.py`, `provider_configuration.py`, `ai_model_configuration.py`
- Updated: `provider.py`, `models/__init__.py`, `core/enums.py`, `schemas/chat.py`, `schemas/health.py`, `services/ollama_service.py`
- Removed: `model.py`
- Docs: ARCHITECTURE, PROJECT_MEMORY, CHANGELOG, ROADMAP, ADR-008

**Decisions:** See [ADR-008](architecture/ADR-008-aimodel-and-configuration-entities.md).
- `Model` → `AIModel` (table `ai_models`)
- Capability ownership: Provider = API; AIModel = this model
- `ProviderConfiguration` / `AIModelConfiguration` as 1:1; JSONB extras; `api_key_env` only
- Runtime enum `Provider` → `ProviderType`

**Next task:** Remaining domain models or Alembic.

### 2026-07-23 — Phase 3.3 Core Domain Models

**Phase:** 3.3

**Objective:** Implement Provider and Model ORM entities with bidirectional relationship. No repositories, migrations, or APIs.

**Files changed:**
- `backend/src/models/provider.py` (created)
- `backend/src/models/model.py` (created; later renamed in 3.4)
- `backend/src/models/__init__.py` (exports updated)
- `docs/ARCHITECTURE.md`, `docs/PROJECT_MEMORY.md`, `docs/CHANGELOG.md`, `docs/ROADMAP.md`

**Decisions:** See [ADR-007](architecture/ADR-007-domain-model-keys-and-provider-identity.md).
- Integer primary keys; string `name` / `provider_type` (not DB ENUM).
- Capability flags at provider and model levels.
- `lazy="selectin"`; unique `(provider_id, model_name)`; FK `ON DELETE CASCADE`.
- ORM `Provider` coexisted with runtime enum `Provider` (renamed to `ProviderType` in 3.4).

**Remaining work:** Other domain entities, Alembic, repositories.

**Next task:** Phase 3.4 configuration entities / refactors.

### 2026-07-20 — Documentation & Engineering Standards Bootstrap

**Phase:** Documentation bootstrap

**Objective:** Populate Cursor rules and project documentation.

**Files:** `.cursor/rules/00`–`08`, docs under `docs/`

### 2026-07-20 — Phase 3.2.1 ORM Foundation

**Phase:** 3.2.1

**Objective:** Create ORM foundation without tables or migrations.

**Files changed:**
- `backend/src/models/base.py`, `mixins.py`, `__init__.py`
- `backend/src/core/base.py` — re-export bridge

**Decisions:**
- `models/base.py` is canonical; `core/base.py` re-exports to prevent duplicate metadata registries.
- Timestamps use `DateTime(timezone=True)` with server defaults.
