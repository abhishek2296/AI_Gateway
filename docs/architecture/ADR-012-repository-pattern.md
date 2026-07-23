# ADR-012: Repository Pattern

## Status

Accepted

## Context

Phase 3.8 introduces the data access layer between services and SQLAlchemy ORM entities. Ten domain models need typed, async-safe persistence queries without duplicating CRUD logic or leaking transaction control into repositories.

## Decision

1. **Package layout** — `backend/src/repositories/` with one module per entity plus `base.py`.
2. **Generic base** — `BaseRepository[ModelT]` provides shared CRUD: `create`, `get_by_id`, `get_one`, `list`, `update`, `delete`, `exists`, `count`.
3. **Query style** — SQLAlchemy 2.x `select()` API only; no raw SQL strings.
4. **Transaction boundary** — repositories may `flush()` and `refresh()` but never `commit()` or `rollback()`. Services or Unit of Work (Phase 3.9) own transactions.
5. **Constructor injection** — each repository accepts `AsyncSession` in `__init__`. FastAPI DI wiring deferred to when services consume repos.
6. **Entity-specific methods** — only persistence queries (filters, ordering, pagination, optional `selectinload` via `options`).
7. **Eager loading** — optional `options` parameter on `list()` / `get_one()`; models already use `lazy="selectin"` for relationship defaults.

## Consequences

### Positive

- Single place for CRUD; entity repos stay thin and focused.
- Type-safe generics; easy to test with injected sessions.
- Compatible with future Unit of Work wrapping one session across repos.

### Negative

- No repository interface ABC yet — concrete classes only (acceptable for internal gateway).
- DI factory functions not added in this phase — services must construct repos manually until wired.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Active Record on models | Violates layer boundaries; models stay persistence-agnostic |
| Commit inside repositories | Prevents coordinated multi-entity transactions |
| Single mega-repository | Violates SRP; hard to maintain |
| Raw SQL for analytics | Premature; ORM queries sufficient for Phase 3.8 |
