# AI Gateway

Production-grade, provider-agnostic HTTP API for routing AI workloads to multiple LLM backends.

Documentation lives under [`docs/`](docs/):

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Changelog](docs/CHANGELOG.md)

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose v2)
- Git

Optional for full chat functionality:

- [Ollama](https://ollama.com/) running on the host (default provider)

---

## Quick Start (Docker)

1. Clone the repository and copy environment variables:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum `POSTGRES_PASSWORD` (and cloud API keys if not using Ollama).

2. Start the **development** stack (API + PostgreSQL, hot reload):

```bash
docker compose -f docker-compose.dev.yml up --build
```

3. Open:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Root: [http://localhost:8000/](http://localhost:8000/)

---

## Development Setup

```bash
docker compose -f docker-compose.dev.yml up --build
```

Features:

- Hot reload (`uvicorn --reload`) with `backend/src` mounted as a volume
- Alembic migrations run automatically on startup
- Environment loaded from `.env`
- Ollama on host reachable via `host.docker.internal` (default in compose)

### Development commands

| Action | Command |
|--------|---------|
| Start (foreground) | `docker compose -f docker-compose.dev.yml up --build` |
| Start (background) | `docker compose -f docker-compose.dev.yml up --build -d` |
| View logs | `docker compose -f docker-compose.dev.yml logs -f api` |
| Rebuild API image | `docker compose -f docker-compose.dev.yml build api` |
| Stop | `docker compose -f docker-compose.dev.yml down` |
| Stop and remove volumes | `docker compose -f docker-compose.dev.yml down -v` |

---

## Production Setup

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Features:

- Multi-stage Docker build (slim runtime image)
- Non-root `appuser` inside the container
- `restart: unless-stopped`
- Built-in health checks (Dockerfile + Compose)
- No source volume mounts

### Production commands

| Action | Command |
|--------|---------|
| Start | `docker compose -f docker-compose.prod.yml up -d --build` |
| Status | `docker compose -f docker-compose.prod.yml ps` |
| Logs | `docker compose -f docker-compose.prod.yml logs -f api` |
| Stop | `docker compose -f docker-compose.prod.yml down` |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_DB` | PostgreSQL database name | `ai_coding_assistant` |
| `POSTGRES_USER` | PostgreSQL user | `ai_app` |
| `POSTGRES_PASSWORD` | PostgreSQL password | *(required)* |
| `POSTGRES_PORT` | Host port for Postgres | `5432` |
| `APP_PORT` | Host port for the API | `8000` |
| `DATABASE_URL` | Async SQLAlchemy URL | Overridden in Compose to use `postgres` host |
| `OLLAMA_HOST` | Ollama base URL | `http://host.docker.internal:11434` in Compose |
| `DEFAULT_PROVIDER` | Default LLM provider | `ollama` |
| `DEFAULT_MODEL` | Default model | `qwen3:8b` |
| `LOG_LEVEL` | Logging level | `INFO` |

See [`.env.example`](.env.example) for the full list.

Compose files inject `DATABASE_URL` with the `postgres` service hostname. You do not need to edit `DATABASE_URL` in `.env` for Docker unless running the API outside Compose.

---

## Postgres Only (Legacy)

The original Postgres-only compose file remains for local pytest / integration tests:

```bash
docker compose up -d
```

---

## Local Development (Without Docker)

Run from `backend/` with [uv](https://docs.astral.sh/uv/):

```bash
cd backend
uv sync --group dev
docker compose up -d   # Postgres only, from repo root
uv run uvicorn src.main:app --reload
```

---

## Project Layout

```
backend/          FastAPI application
docs/             Architecture, roadmap, ADRs
docker-compose.dev.yml    Development stack (API + Postgres)
docker-compose.prod.yml   Production stack (API + Postgres)
docker-compose.yml        Postgres only (legacy)
```
