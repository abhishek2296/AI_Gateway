# ADR-007: Integer PKs and String Provider Identity for Domain Models

## Status

Accepted

## Context

Phase 3.3 introduces the first persisted domain entities: `Provider` and `Model`. Decisions were required for primary key type, how to identify providers in the database, and relationship loading strategy under async SQLAlchemy.

The runtime layer already has `src.core.enums.Provider` (a small string enum used in API responses). Expanding that enum into the ORM would couple schema migrations to every new backend.

## Decision

1. Use **integer** primary keys (`Mapped[int]`) for `Provider` and `Model`. The repository has no UUID convention yet; the database Cursor rule examples use integers.
2. Store provider identity as **strings** (`name`, `provider_type`) with a unique index on `name`, not as a SQLAlchemy/PostgreSQL ENUM type.
3. Keep the runtime enum `src.core.enums.Provider` unchanged for API responses; treat `src.models.Provider` as a separate ORM type.
4. Use `lazy="selectin"` for both sides of the `Provider` ↔ `Model` relationship so async sessions do not trigger implicit lazy IO (`MissingGreenlet`).
5. Enforce uniqueness of `(provider_id, model_name)` and cascade-delete models when a provider is removed.

## Consequences

### Positive

- New providers can be inserted as rows without altering a DB enum or redeploying enum members.
- Integer PKs keep joins and indexes simple for early schema.
- `selectin` loading is safe with `AsyncSession`.

### Negative

- Integer IDs are easier to guess than UUIDs (acceptable while there is no public multi-tenant API surface).
- Two types named `Provider` (enum vs ORM) require careful imports.
- `is_default` uniqueness per provider is not enforced at the DB level yet (application concern for a later phase).

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| UUID primary keys | No existing UUID pattern; adds complexity without current multi-tenant need |
| PostgreSQL ENUM for `provider_type` | Requires migrations for each new provider type; fights “minimal change to add a provider” |
| Reuse / merge runtime enum into ORM | Couples HTTP DTOs to persistence; would force renaming or breaking API enums now |
| `lazy="select"` (default) | Unsafe with async SQLAlchemy without explicit `await` loading |
| Soft-delete only (no CASCADE) | Cascade is appropriate for catalog ownership; soft-delete can be added later if needed |
