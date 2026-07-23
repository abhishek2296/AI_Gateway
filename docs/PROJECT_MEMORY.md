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
│       └── 20260723_2200_gateway_operational_models.py
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

**10 ORM entities** + **11 repository classes** + **Unit of Work** + **38 integration tests**.

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
| 3.10 — Persistence Layer Testing | pytest suite, 38 integration tests |

---

## Current Work

**Next:** **Phase 4 — Multi-Provider Architecture**

---

## Remaining Work (Phase 3)

None — Phase 3 complete.

---

## Memory Log

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
