# ADR-013: Unit of Work Pattern

## Status

Accepted

## Context

Phase 3.8 introduced ten entity repositories sharing an `AsyncSession`. Services need a single place to coordinate multi-entity writes and own `commit` / `rollback` / `close` without repositories managing transactions.

## Decision

1. **Package** — `backend/src/unit_of_work/` with `BaseUnitOfWork` (ABC) and `AsyncUnitOfWork` (concrete).
2. **Repository registration** — `AsyncUnitOfWork` instantiates all ten repositories with one session and exposes them as properties (`providers`, `ai_models`, etc.).
3. **Transaction ownership** — only the Unit of Work calls `session.commit()`, `session.rollback()`, and (optionally) `session.close()`.
4. **Context manager** — `BaseUnitOfWork.__aexit__` rolls back on exception and calls `close()`; **never auto-commits**.
5. **Session lifecycle** — `close_session=True` by default; set `close_session=False` when an outer context (e.g. FastAPI `get_session`) owns the session.
6. **No DI in this phase** — services will construct or receive UoW in a later phase.

## Consequences

### Positive

- Atomic multi-repository operations via explicit `await uow.commit()`.
- Clear separation: repositories persist, UoW coordinates transactions.
- Extensible ABC for future test doubles or sync variants.

### Negative

- One UoW instance creates all repositories even if only one is used (acceptable — lightweight).
- Callers must remember explicit commit; uncommitted work is rolled back on close.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Commit inside repositories | Breaks atomic multi-entity transactions |
| Auto-commit on context exit | Hides transaction boundaries; risky for partial failures |
| Service-layer commit only | Duplicates session coordination across services |
| Generic repository registry dict | Less type-safe than named properties |
