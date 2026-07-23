# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- Engineering standards bootstrap: Cursor rules (`.cursor/rules/00`–`08`)
- Project documentation under `docs/`
- Architecture Decision Records (ADR stubs / docs)

---

## Phase 3.11 — Persistence Hardening (2026-07-23)

### Added

- Migration `c8f5e2a31d04` — unique `usage_records.request_id`, partial unique indexes for one default `AIModel` and `APIKey` per provider
- Integration tests for duplicate `request_id` and duplicate default model/key rejection

### Changed

- `backend/src/models/usage_record.py`, `ai_model.py`, `api_key.py` — constraint definitions
- `docs/ARCHITECTURE.md`, `docs/ROADMAP.md` — Phase 3 completion and constraint documentation

---

## Phase 3.10 — Persistence Layer Testing (2026-07-23)

### Added

- `backend/tests/` — pytest integration suite (38 tests) against PostgreSQL
- `conftest.py` — test DB creation, Alembic setup, transactional isolation fixtures
- `factories.py`, `helpers.py` — entity builders and IntegrityError helper
- Integration tests: Alembic, ORM models, all repositories, Unit of Work
- Dev dependencies: `pytest`, `pytest-asyncio` in `[dependency-groups] dev`
- `TEST_DATABASE_URL` setting in `core/config.py` and `.env.example`

### Changed

- `backend/pyproject.toml` — pytest configuration
- `docs/ARCHITECTURE.md` — testing section
- `.cursor/rules/07-testing.mdc` — updated from planned to implemented

---

## Phase 3.9 — Unit of Work (2026-07-23)

### Added

- `backend/src/unit_of_work/base.py` — `BaseUnitOfWork` ABC with async context manager
- `backend/src/unit_of_work/unit_of_work.py` — `AsyncUnitOfWork` with all 10 repository properties
- [ADR-013](architecture/ADR-013-unit-of-work-pattern.md)

### Changed

- `backend/src/repositories/base.py` — docstring references UoW for transaction ownership
- `docs/ARCHITECTURE.md` — Unit of Work section, layer boundaries
- `.cursor/rules/04-database.mdc` — UoW transaction conventions

---

## Phase 3.8 — Repository Pattern (2026-07-23)

### Added

- `backend/src/repositories/base.py` — `BaseRepository[ModelT]` with async CRUD, pagination, filtering, ordering
- Entity repositories for all 10 ORM models (`ProviderRepository` through `ProviderHealthRepository`)
- [ADR-012](architecture/ADR-012-repository-pattern.md)

### Changed

- `docs/ARCHITECTURE.md` — repository layer section, layer boundaries table
- `.cursor/rules/04-database.mdc` — repository session usage conventions

---

## Phase 3.7 — Gateway Operational Models (2026-07-23)

### Added

- `backend/src/models/api_key.py` — `APIKey` (env-var credential references, unique `(provider_id, name)`)
- `backend/src/models/usage_record.py` — `UsageRecord` (token counts, cost, latency, billing audit)
- `backend/src/models/provider_health.py` — `ProviderHealth` (historical health snapshots)
- `backend/alembic/versions/20260723_2200_gateway_operational_models.py` — migration `b7e4d9f21c03`
- [ADR-011](architecture/ADR-011-gateway-operational-models.md)

### Changed

- `backend/src/models/provider.py` — `api_keys`, `usage_records`, `health_checks` relationships
- `backend/src/models/ai_model.py` — `usage_records` relationship
- `backend/src/models/chat_session.py` — `usage_records` relationship
- `backend/src/models/__init__.py` — exports all 10 ORM entities
- `docs/ARCHITECTURE.md` — ER diagram extended; migrations table updated

---

## Phase 3.6 — Alembic Migration Infrastructure (2026-07-23)

### Added

- `backend/alembic.ini` — Alembic config (`prepend_sys_path`, sortable `file_template`, `revision_environment`)
- `backend/alembic/env.py` — async migration env (`asyncpg`, `Base.metadata`, `settings.DATABASE_URL`)
- `backend/alembic/script.py.mako` — revision template
- `backend/alembic/versions/20260723_2108_initial_schema.py` — initial migration (`a3f6c2d18e01`) for 7 domain tables
- `backend/alembic/README` — migration command reference
- [ADR-010](architecture/ADR-010-alembic-async-migrations.md)
- Migrations section in `docs/ARCHITECTURE.md`

### Changed

- `.cursor/rules/04-database.mdc` — Alembic workflow documented

---

## Phase 3.5 — Conversation Domain Models (2026-07-23)

### Added

- `backend/src/models/chat_session.py` — `ChatSession` (UUID, provider/model FKs, session overrides, JSONB metadata)
- `backend/src/models/message.py` — `Message` (role, content, metrics, provider response metadata)
- `backend/src/models/prompt_template.py` — `PromptTemplate` (versioned templates, JSONB variables)
- Bidirectional relationships: `Provider`/`AIModel` → `ChatSession` → `Message`
- [ADR-009](architecture/ADR-009-conversation-domain-models.md)

### Changed

- `backend/src/models/provider.py` — `chat_sessions` relationship
- `backend/src/models/ai_model.py` — `chat_sessions` relationship
- `backend/src/models/__init__.py` — exports new entities
- `docs/ARCHITECTURE.md` — extended ER diagram for conversation domain

---

## Phase 3.4 — Provider & Model Configuration (2026-07-23)

### Added

- `backend/src/models/ai_model.py` — `AIModel` ORM entity (renamed from `Model`; table `ai_models`)
- `backend/src/models/provider_configuration.py` — 1:1 `ProviderConfiguration` (connection settings, JSONB `extra_config`)
- `backend/src/models/ai_model_configuration.py` — 1:1 `AIModelConfiguration` (generation defaults, JSONB `extra_parameters`)
- Nullable `description` on `Provider` and `AIModel`
- [ADR-008](architecture/ADR-008-aimodel-and-configuration-entities.md)

### Changed

- Removed ORM entity named `Model` / file `model.py`
- `Provider.ai_models` replaces `Provider.models`; capability ownership documented on entities
- Runtime enum `core.enums.Provider` renamed to `ProviderType` (schemas + `OllamaService` updated)
- Deduplicated `ChatResponse` in `schemas/chat.py` while updating the enum reference
- `docs/ARCHITECTURE.md` ER diagram updated for configuration entities

### Removed

- `backend/src/models/model.py`

---

## Phase 3.3 — Core Domain Models (2026-07-23)

### Added

- `backend/src/models/provider.py` — `Provider` ORM entity (slug, capabilities, local/active flags)
- `backend/src/models/model.py` — `Model` ORM entity (per-provider model catalog, token limits, capability flags)
- Bidirectional `Provider.models` ↔ `Model.provider` relationship with `lazy="selectin"` and cascade delete
- Unique constraint `uq_models_provider_id_model_name` on `(provider_id, model_name)`
- Database design section and ER diagram in `docs/ARCHITECTURE.md`

### Changed

- `backend/src/models/__init__.py` — exports `Provider` and `Model`
- `docs/ROADMAP.md` — Phase 3.3 marked complete; next work noted

---

## Phase 3.2.1 — ORM Foundation (2026-07-20)

### Added

- `backend/src/models/base.py` — single `DeclarativeBase` registry for all ORM models
- `backend/src/models/mixins.py` — `TimestampMixin` with `created_at` and `updated_at` (SQLAlchemy 2.x typed ORM)
- `backend/src/models/__init__.py` — public exports for `Base` and `TimestampMixin`

### Changed

- `backend/src/core/base.py` — re-exports `Base` from `src.models.base` to maintain a single metadata registry

---

## Phase 3.1 — Database Infrastructure (2026-07-20)

### Added

- `backend/src/core/database.py` — async SQLAlchemy engine and `AsyncSessionLocal` session factory
- `backend/src/core/session.py` — `get_session()` FastAPI dependency
- `docker-compose.yml` — PostgreSQL 17 with health check and named volume
- `.env.example` — database connection variables (`DATABASE_URL`, pool settings)
- Database settings in `core/config.py` (`DATABASE_URL`, `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_ECHO`)
- Engine disposal on shutdown via `lifespan.py`

### Dependencies

- `sqlalchemy>=2.0`, `asyncpg>=0.29`, `alembic>=1.13` added to `pyproject.toml`

---

## Phase 2 — AI Gateway Core (2026-07-20)

### Added

- Provider abstraction: `BaseLLMService` ABC (`services/base_llm.py`)
- Ollama provider: `OllamaService` with chat and connection check
- Chat orchestration: `ChatService`
- `POST /chat` — send message, receive normalized LLM response
- `GET /health` — provider connectivity check with latency measurement
- Pydantic schemas: `ChatRequest`, `ChatResponse`, `HealthResponse`
- Generic response envelope: `APIResponse[T]`, `FailureResponse`, `ErrorResponse`
- Custom exceptions: `OllamaConnectionException` (503), `LLMResponseException` (500)
- Global HTTP exception handler with standardized error JSON
- Application lifespan: Ollama connectivity check on startup
- Enums: `Provider`, `HealthStatus`
- FastAPI dependency injection in `api/dependencies.py`
- Manual smoke test script: `backend/test.py`

---

## Phase 1 — Project Foundation (2026-07-20)

### Added

- FastAPI application entry point (`main.py`)
- Pydantic settings with env file loading (`core/config.py`)
- Structured logging setup (`core/logging.py`)
- `RequestIDMiddleware` — UUID per request, `X-Request-ID` header
- `TimingMiddleware` — request duration logging, `X-Response-Time` header
- `GET /` root status endpoint
- Python 3.14 project with `uv` package management (`pyproject.toml`, `uv.lock`)
- `.gitignore` for Python, venv, and `.env`
- Docker Compose service definition for PostgreSQL

---

## [0.1.0] — Initial Release

First working AI Gateway with Ollama chat and health endpoints, async PostgreSQL infrastructure, and ORM foundation.
