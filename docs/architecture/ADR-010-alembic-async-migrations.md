# ADR-010: Async Alembic Migrations

## Status

Accepted

## Context

Phase 3.6 requires database schema versioning for seven ORM tables. The application uses async SQLAlchemy with `postgresql+asyncpg://`. Alembic must share the single `Base.metadata` registry and load `DATABASE_URL` from application settings.

## Decision

1. **Location:** `backend/alembic/` + `backend/alembic.ini`; all commands run from `backend/`.
2. **Async env:** `alembic/env.py` uses `async_engine_from_config` + `connection.run_sync(do_run_migrations)` — no sync driver (`psycopg2`) added.
3. **Metadata discovery:** `import src.models` side-effect registers all tables; `target_metadata = Base.metadata`.
4. **URL source:** `settings.DATABASE_URL` from `src.core.config` (never hardcoded in `alembic.ini`).
5. **Autogenerate options:** `compare_type=True`, `compare_server_default=True`.
6. **Initial migration:** `a3f6c2d18e01` creates all 7 domain tables; reviewed against ORM metadata (including `metadata` JSONB columns on sessions/messages).
7. **Naming:** `file_template` uses date-time prefix + slug for sortable revision filenames.
8. **No manual DDL** outside Alembic going forward.

## Consequences

### Positive

- Schema changes are versioned and reproducible.
- Same async driver stack as the application.
- Single metadata registry avoids drift between ORM and migrations.

### Negative

- Requires PostgreSQL running and valid `.env` for live `upgrade`/`autogenerate`.
- Async env is slightly more complex than sync Alembic templates.
- Initial migration was metadata-reviewed (offline SQL verified) when live autogenerate was blocked by DB credentials.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Sync Alembic + `psycopg2` | Extra dependency; app already uses asyncpg |
| Alembic at repo root | `src.*` imports cleaner from `backend/` |
| `create_all()` without Alembic | No downgrade path; not production-ready |
| Hardcoded URL in `alembic.ini` | Violates secrets/config conventions |
