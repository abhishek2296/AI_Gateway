# ADR-009: Conversation Domain Models

## Status

Accepted

## Context

Phase 3.5 adds persistence for conversations: `ChatSession`, `Message`, and `PromptTemplate`. Design choices were needed for external identifiers, metadata column naming, provider/model denormalization, message ordering, and template versioning.

Existing patterns from ADR-007 and ADR-008: integer PKs, string identities (not DB ENUMs), JSONB for extensions, `lazy="selectin"` for async SQLAlchemy.

## Decision

1. **`ChatSession.session_uuid`** — PostgreSQL `UUID` with unique index; stable public identifier separate from integer PK.
2. **Denormalized `provider_id` on `ChatSession`** — alongside `ai_model_id` for fast provider-scoped queries; service layer validates `ai_model.provider_id == provider_id` (no DB check constraint yet).
3. **`extra_metadata` Python attribute** — maps to DB column `metadata` (JSONB) on `ChatSession` and `Message`; avoids shadowing SQLAlchemy declarative `Base.metadata`.
4. **`Message.role` as string** — values like `system`, `user`, `assistant`, `tool`; no DB ENUM (consistent with ADR-007). Runtime `MessageRole` enum deferred to API layer.
5. **Message ordering** — by `created_at` via relationship `order_by`; no explicit `sequence` column in this phase.
6. **`PromptTemplate` standalone** — no FK to `ChatSession`; unique on `(name, version)` for versioned templates.
7. **Cascade delete** — deleting a `ChatSession` cascades to its `Message` rows; FKs to `providers` and `ai_models` use `ON DELETE CASCADE`.

## Consequences

### Positive

- Conversation history is persistable without APIs or repositories yet.
- External clients can reference sessions by UUID.
- Template versioning preserves history without overwriting.
- Consistent with established ORM conventions.

### Negative

- Denormalized `provider_id` requires application-level consistency checks.
- `extra_metadata` diverges slightly from spec field label `metadata`.
- No template→session linkage until a future API phase.
- Concurrent message inserts rely on `created_at` ordering (may need explicit sequence later).

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| UUID primary keys for all entities | No existing UUID convention; integer PKs sufficient for internal joins |
| DB check constraint for provider/model consistency | Adds complexity; defer to service layer |
| Python attr named `metadata` | Collides conceptually with SQLAlchemy metadata |
| DB ENUM for message role | Requires migrations for new roles; conflicts with ADR-007 |
| Explicit `sequence` on Message | Premature; `created_at` sufficient for now |
| FK from ChatSession to PromptTemplate | Out of scope; templates are catalog-only in this phase |
