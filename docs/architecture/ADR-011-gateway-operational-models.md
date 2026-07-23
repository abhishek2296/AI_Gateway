# ADR-011: Gateway Operational Models

## Status

Accepted

## Context

Phase 3.7 completes the ORM layer with operational entities: `APIKey`, `UsageRecord`, and `ProviderHealth`. Design choices were needed for secret storage, billing record retention, and health check history.

## Decision

1. **APIKey** — store `api_key_env` (env var name) only; never persist secret values. `key_identifier` is a non-secret admin label. Unique `(provider_id, name)`.
2. **UsageRecord** — immutable analytics/billing row per request. `request_id` correlates with `X-Request-ID`. `estimated_cost` uses `Numeric(12, 6)`.
3. **FK delete semantics on UsageRecord:**
   - `chat_session_id` → `SET NULL` (preserve billing when session removed)
   - `provider_id` / `ai_model_id` → `RESTRICT` (prevent catalog deletion with billing history)
4. **ProviderHealth** — one-to-many historical snapshots (not 1:1); latest by `checked_at`. `status` as string per ADR-007.
5. **Relationships** — all use `lazy="selectin"` for async compatibility.

## Consequences

### Positive

- Secrets stay in environment variables; DB holds references only.
- Usage records survive session deletion for billing/analytics.
- Provider/model deletion blocked while usage records exist.
- Health history enables trend analysis.

### Negative

- Providers/models with usage records cannot be deleted without archival strategy.
- `ProviderConfiguration.api_key_env` and `APIKey.api_key_env` may overlap — configuration vs credential registry serve different phases (config defaults vs multi-key rotation).

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Store hashed API keys in DB | Violates security posture; env vars sufficient for now |
| CASCADE delete usage records | Loses billing/audit trail |
| 1:1 ProviderHealth | No historical trend data |
| Float for estimated_cost | Precision loss for billing |
