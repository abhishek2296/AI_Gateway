# ADR-008: AIModel Rename, ProviderType Enum, and Configuration Entities

## Status

Accepted

## Context

Phase 3.3 introduced `Provider` and `Model` ORM entities. Review feedback identified:

1. An ORM class named `Model` collides conceptually with SQLAlchemy/Pydantic "model" terminology.
2. Capability flags needed clearer ownership (provider API vs specific model).
3. Connection settings and generation defaults should not live on catalog rows.
4. The runtime enum `core.enums.Provider` collided in name with the ORM `Provider`.

## Decision

1. Rename ORM `Model` → **`AIModel`** (table `ai_models`). Relationship on Provider is `ai_models`.
2. Document capability ownership in entity docstrings and architecture docs:
   - **Provider** — API-level capabilities
   - **AIModel** — per-model capabilities
3. Add nullable `description` on `Provider` and `AIModel`.
4. Introduce **`ProviderConfiguration`** (1:1 with Provider) for connection settings; store secrets only as `api_key_env` (env var name).
5. Introduce **`AIModelConfiguration`** (1:1 with AIModel) for default generation parameters.
6. Use PostgreSQL **JSONB** for `extra_config` / `extra_parameters` extension bags.
7. Rename runtime enum `Provider` → **`ProviderType`** and update API schemas/services (feasible; few call sites).
8. Enforce 1:1 via unique FK columns + `uselist=False` + `lazy="selectin"`.

## Consequences

### Positive

- Clearer naming; no ORM entity called `Model`.
- Catalog rows stay lean; config is separable and optional (`configuration` may be `None`).
- Secrets stay out of the database.
- Runtime enum no longer shares a name with the ORM entity.

### Negative

- Table rename `models` → `ai_models` will matter once Alembic exists (no migrations yet, so cost is zero today).
- JSONB ties extension columns to PostgreSQL (acceptable; production DB is Postgres).
- Two configuration tables add join surface for future repositories.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Keep class name `Model` | Ambiguous in ORM/Pydantic contexts |
| Embed config columns on Provider/AIModel | Mixes catalog identity with connection/runtime knobs; harder to evolve |
| Store API keys in DB | Violates security posture; use env var names instead |
| Defer enum rename | Only three call sites; renaming now prevents ongoing confusion |
| Generic `JSON` instead of `JSONB` | Less efficient on PostgreSQL; no multi-DB requirement today |
