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
| Migrations (planned) | Alembic |

---

## Folder Structure (models)

```
backend/src/models/
├── base.py
├── mixins.py
├── provider.py
├── provider_configuration.py
├── ai_model.py
├── ai_model_configuration.py
└── __init__.py
```

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

---

## Current Work

**Next:** Remaining domain models (ChatSession, Message, PromptTemplate, APIKey) or Alembic setup.

---

## Remaining Work (Phase 3)

- [ ] Additional domain models (ChatSession, Message, PromptTemplate, APIKey)
- [ ] Alembic setup + initial migration
- [ ] Repository pattern
- [ ] Unit of Work
- [ ] Automated tests

---

## Memory Log

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
