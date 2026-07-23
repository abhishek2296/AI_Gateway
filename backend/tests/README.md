# Persistence integration tests

Run from `backend/`:

```bash
docker compose up -d
uv run pytest
```

Requires PostgreSQL (see `TEST_DATABASE_URL` in `.env.example`). Tests use a separate database and roll back after each case.
