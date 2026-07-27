# AI Gateway Roadmap

## Project Vision

Build a **production-grade, provider-agnostic AI Gateway** — a reusable HTTP API that routes AI workloads (chat, embeddings, vision, RAG, agents, function calling) to multiple LLM backends with minimal code changes per provider.

The gateway is **infrastructure**, not a chatbot or coding assistant.

## Development Principles

- Learn before implementing
- Review before coding
- Extend, don't rewrite
- Production-first architecture
- Documentation is part of Definition of Done

## High-Level Milestones

| Milestone | Target Capability |
|-----------|-------------------|
| M1 — Gateway Core | Chat + health via first provider | ✅ |
| M2 — Persistence | PostgreSQL, ORM, migrations, repositories | ✅ |
| M3 — Multi-Provider | OpenAI, Anthropic, Gemini, Azure, Bedrock | Planned |
| M4 — Routing & Streaming | Model registry, intelligent routing, SSE | Planned |
| M5 — Enterprise | Auth, rate limits, observability, K8s | Planned |

---

## Current Phase

**Phase 3 — Persistence Layer** (complete)

Completed: **Phase 4 — Multi-Provider Architecture**, **Phase 5 — Cloud Provider Implementations** (including 5.8 Dockerization)

Next: **Phase 6 — Model Registry**

---

## Phase Details

### Phase 1 — Project Foundation ✅

**Deliverables:**
- FastAPI application bootstrap (`main.py`)
- Pydantic settings (`core/config.py`)
- Structured logging (`core/logging.py`)
- Request ID + timing middleware
- Global exception handler
- Docker Compose (PostgreSQL 17)
- Environment variable template (`.env.example`)
- `uv` dependency management

### Phase 2 — AI Gateway Core ✅

**Deliverables:**
- Provider abstraction (`BaseLLMService`)
- Ollama provider implementation
- Chat service orchestration
- `POST /chat` endpoint
- `GET /health` endpoint (provider connectivity + latency)
- `GET /` root status endpoint
- Pydantic schemas with `APIResponse[T]` envelope
- Application lifespan (startup health check, shutdown DB cleanup)
- Custom exceptions (`OllamaConnectionException`, `LLMResponseException`)

### Phase 3 — Persistence Layer ✅

| Sub-phase | Status | Deliverables |
|-----------|--------|-------------|
| 3.1 Database Infrastructure | ✅ | Async SQLAlchemy engine, session factory, `get_session()` |
| 3.2.1 ORM Foundation | ✅ | `Base`, `TimestampMixin`, `models/__init__.py` |
| 3.3 Core Domain Models | ✅ | `Provider`, `AIModel` (originally `Model`) + relationship |
| 3.4 Provider & Model Configuration | ✅ | Refactors, `ProviderConfiguration`, `AIModelConfiguration` |
| 3.5 Conversation Domain Models | ✅ | `ChatSession`, `Message`, `PromptTemplate` |
| 3.6 Alembic | ✅ | Async env, initial migration (`a3f6c2d18e01`) |
| 3.7 Gateway Operational Models | ✅ | `APIKey`, `UsageRecord`, `ProviderHealth` + migration `b7e4d9f21c03` |
| 3.8 Repository Pattern | ✅ | `BaseRepository` + 10 entity repositories |
| 3.9 Unit of Work | ✅ | `BaseUnitOfWork`, `AsyncUnitOfWork` |
| 3.10 Testing | ✅ | pytest + PostgreSQL integration suite |
| 3.11 Hardening | ✅ | Unique `request_id`, one default per provider (model/key) |

### Phase 4 — Multi-Provider Architecture 🚧

| Sub-phase | Status | Deliverables |
|-----------|--------|-------------|
| 4.1 Provider Abstraction | ✅ | `BaseProvider`, exception hierarchy, normalized DTOs |
| 4.2 Provider Registry & Factory | ✅ | `ProviderRegistry`, `ProviderFactory`, `register_provider` |
| 4.3 Ollama Adapter | ✅ | `OllamaProvider`, REST mapping, respx unit tests |
| 4.4 Service Integration | ✅ | `AIService`, factory DI, legacy adapter |
| 4.5 Database Resolution | ✅ | `ProviderResolver`, config resolver, TTL cache |
| 4.6 Provider Skeletons | ✅ | OpenAI, Anthropic, Gemini skeletons; `HTTPProviderMixin` |

**Phase 4 complete.** Cloud vendor API integration moved to Phase 5.

**Phase 4.1 deliverables:**
- `backend/src/providers/` — `BaseProvider` ABC, DTOs, `ProviderError` hierarchy
- [ADR-003](architecture/ADR-003-provider-abstraction.md)

**Phase 4.2 deliverables:**
- `backend/src/providers/registry.py` — thread-safe `ProviderRegistry`, `register_provider`, `get_registry`
- `backend/src/providers/factory.py` — `ProviderFactory` (no instance caching)
- Unit tests in `backend/tests/unit/providers/`

**Phase 4.3 deliverables:**
- `backend/src/providers/ollama.py` — `OllamaProvider` with httpx REST adapter
- respx unit tests; auto-registration on import
- [ADR-014](architecture/ADR-014-ollama-provider.md)

**Phase 4.4 deliverables:**
- `backend/src/services/ai_service.py` — provider-agnostic orchestration
- `backend/src/services/llm_adapter.py` — health route backward compatibility
- Unit tests in `backend/tests/unit/services/`
- [ADR-015](architecture/ADR-015-provider-service-integration.md)

**Phase 4.5 deliverables:**
- `ProviderResolver`, `ProviderConfigResolver`, `ProviderResolutionCoordinator`
- `ProviderRepository.get_default_provider()`, TTL cache in `core/`
- [ADR-016](architecture/ADR-016-provider-resolution.md)

**Phase 4.6 deliverables:**
- `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider` skeletons
- `HTTPProviderMixin`; Ollama refactored to use mixin
- Unit tests in `backend/tests/unit/providers/test_provider_skeletons.py`
- [ADR-017](architecture/ADR-017-provider-skeletons.md)

**Planned capabilities (future phases):**
- Azure OpenAI and AWS Bedrock adapters
- Provider-specific routing policies

### Phase 5 — Cloud Provider Implementations ✅

| Milestone | Status | Deliverables |
|-----------|--------|-------------|
| 5.0 Foundation | ✅ | Extended DTOs, shared HTTP modules, Ollama error refactor |
| 5.1 OpenAI Core | ✅ | Chat, list_models, health_check |
| 5.2 OpenAI Advanced | ✅ | Streaming, vision, tools, JSON |
| 5.3 OpenAI Embeddings | ✅ | Embeddings, validate_model |
| 5.4 Anthropic | ✅ | Messages API adapter |
| 5.5 Gemini | ✅ | Generate Content adapter |
| 5.6 Parity | ✅ | Contract tests, `ProviderType` enum |
| 5.7 Documentation | ✅ | ADR-018/019, provider guides |
| 5.8 Dockerization | ✅ | Dockerfile, dev/prod Compose, README |

**Deliverables:**
- Full `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider` (httpx REST)
- Shared: `http_errors`, `http_auth`, `retry`, `streaming`, `validation`
- 115 provider unit tests (respx)
- [ADR-018](architecture/ADR-018-extended-provider-dtos.md), [ADR-019](architecture/ADR-019-cloud-provider-implementations.md)
- Provider configuration guides in `docs/providers/`

**Phase 5.8 deliverables:**
- `backend/Dockerfile` — multi-target build (dev hot reload, prod non-root)
- `docker-compose.dev.yml`, `docker-compose.prod.yml` — full stack (API + Postgres)
- `backend/docker/entrypoint.sh` — wait for DB, Alembic migrate, exec uvicorn
- Root `README.md` — Docker quick start and env reference

### Phase 6 — Model Registry

- Database-backed model catalog
- Capability metadata (chat, embed, vision)
- Default model per provider

### Phase 7 — Routing Engine

- Route requests by model, cost, latency, or policy
- Fallback chains across providers

### Phase 8 — Streaming Engine

- Server-Sent Events (SSE) for chat completions
- Provider-agnostic streaming interface

### Phase 9 — Authentication & Security

- API key management
- JWT / OAuth support
- Rate limiting and tenant isolation

### Phase 10 — Redis

- Response caching
- Session state
- Rate limit counters

### Phase 11 — Vector Database (Qdrant)

- Embedding storage and retrieval
- Collection management

### Phase 12 — RAG

- Document ingestion pipeline
- Retrieval-augmented generation endpoints

### Phase 13 — MCP

- Model Context Protocol integration

### Phase 14 — Function Calling

- Tool definition and execution framework

### Phase 15 — Agents

- Multi-step agent orchestration

### Phase 16 — Workflow Engine

- Composable AI workflow definitions

### Phase 17 — Observability

- Prometheus metrics, Grafana dashboards
- Distributed tracing, structured audit logs

### Phase 18 — Performance

- Connection pooling tuning, caching strategies
- Load testing and benchmarking

### Phase 19 — Production Infrastructure

- Nginx reverse proxy, TLS termination
- Health checks, graceful shutdown

### Phase 20 — CI/CD

- GitHub Actions: lint, test, build, deploy

### Phase 21 — Kubernetes

- Helm charts, HPA, secrets management

### Phase 22 — Cloud Deployment

- AWS / GCP / Azure deployment guides

### Phase 23 — SDKs

- Python and TypeScript client libraries

### Phase 24 — Enterprise Features

- Multi-tenancy, billing, admin dashboard

---

## Completed Phases Summary

| Phase | Completed |
|-------|-----------|
| 1 — Project Foundation | ✅ |
| 2 — AI Gateway Core | ✅ |
| 3.1 — Database Infrastructure | ✅ |
| 3.2.1 — ORM Foundation | ✅ |
| 3.3 — Core Domain Models (Provider, AIModel) | ✅ |
| 3.4 — Provider & Model Configuration | ✅ |
| 3.5 — Conversation Domain Models | ✅ |
| 3.6 — Alembic Migration Infrastructure | ✅ |
| 3.7 — Gateway Operational Models | ✅ |
| 3.8 — Repository Pattern | ✅ |
| 3.9 — Unit of Work | ✅ |
| 3.10 — Persistence Layer Testing | ✅ |
| 3.11 — Persistence Hardening | ✅ |
| 4 — Multi-Provider Architecture | ✅ |
| 5 — Cloud Provider Implementations | ✅ |
| 5.8 — Dockerization | ✅ |

---

## Upcoming (Next 3 Tasks)

1. **Phase 6** — Model registry
2. **Phase 7** — Intelligent routing
3. **Phase 8** — Gateway streaming (SSE at HTTP layer)
