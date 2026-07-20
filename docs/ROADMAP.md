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
| M2 — Persistence | PostgreSQL, ORM, migrations, repositories | 🚧 |
| M3 — Multi-Provider | OpenAI, Anthropic, Gemini, Azure, Bedrock | Planned |
| M4 — Routing & Streaming | Model registry, intelligent routing, SSE | Planned |
| M5 — Enterprise | Auth, rate limits, observability, K8s | Planned |

---

## Current Phase

**Phase 3 — Persistence Layer** (in progress)

Next sub-phase: **3.2.2 — Domain Models**

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

### Phase 3 — Persistence Layer 🚧

| Sub-phase | Status | Deliverables |
|-----------|--------|-------------|
| 3.1 Database Infrastructure | ✅ | Async SQLAlchemy engine, session factory, `get_session()` |
| 3.2.1 ORM Foundation | ✅ | `Base`, `TimestampMixin`, `models/__init__.py` |
| 3.2.2 Domain Models | Planned | Provider, Model, ChatSession, Message, PromptTemplate, APIKey, ProviderConfiguration |
| 3.3 Relationships | Planned | Foreign keys, `relationship()` mappings |
| 3.4 Alembic | Planned | Migration config, initial schema |
| 3.5 Repository Pattern | Planned | Data access layer per entity |
| 3.6 Unit of Work | Planned | Transaction boundary management |
| 3.7 Testing | Planned | pytest, async fixtures, API + DB tests |

### Phase 4 — Multi-Provider Architecture

- Provider registry and factory
- OpenAI, Anthropic, Gemini implementations
- Azure OpenAI and AWS Bedrock adapters
- Provider-specific configuration and error mapping
- Unified response normalization

### Phase 5 — Model Registry

- Database-backed model catalog
- Capability metadata (chat, embed, vision)
- Default model per provider

### Phase 6 — Routing Engine

- Route requests by model, cost, latency, or policy
- Fallback chains across providers

### Phase 7 — Streaming Engine

- Server-Sent Events (SSE) for chat completions
- Provider-agnostic streaming interface

### Phase 8 — Authentication & Security

- API key management
- JWT / OAuth support
- Rate limiting and tenant isolation

### Phase 9 — Redis

- Response caching
- Session state
- Rate limit counters

### Phase 10 — Vector Database (Qdrant)

- Embedding storage and retrieval
- Collection management

### Phase 11 — RAG

- Document ingestion pipeline
- Retrieval-augmented generation endpoints

### Phase 12 — MCP

- Model Context Protocol integration

### Phase 13 — Function Calling

- Tool definition and execution framework

### Phase 14 — Agents

- Multi-step agent orchestration

### Phase 15 — Workflow Engine

- Composable AI workflow definitions

### Phase 16 — Observability

- Prometheus metrics, Grafana dashboards
- Distributed tracing, structured audit logs

### Phase 17 — Performance

- Connection pooling tuning, caching strategies
- Load testing and benchmarking

### Phase 18 — Production Infrastructure

- Nginx reverse proxy, TLS termination
- Health checks, graceful shutdown

### Phase 19 — CI/CD

- GitHub Actions: lint, test, build, deploy

### Phase 20 — Kubernetes

- Helm charts, HPA, secrets management

### Phase 21 — Cloud Deployment

- AWS / GCP / Azure deployment guides

### Phase 22 — SDKs

- Python and TypeScript client libraries

### Phase 23 — Enterprise Features

- Multi-tenancy, billing, admin dashboard

---

## Completed Phases Summary

| Phase | Completed |
|-------|-----------|
| 1 — Project Foundation | ✅ |
| 2 — AI Gateway Core | ✅ |
| 3.1 — Database Infrastructure | ✅ |
| 3.2.1 — ORM Foundation | ✅ |

---

## Upcoming (Next 3 Tasks)

1. **3.2.2** — Implement domain ORM models
2. **3.3** — Define model relationships
3. **3.4** — Configure Alembic and initial migration
