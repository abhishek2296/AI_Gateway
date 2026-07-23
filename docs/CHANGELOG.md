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
