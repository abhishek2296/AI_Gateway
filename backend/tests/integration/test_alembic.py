"""
Alembic migration and schema alignment tests.

These tests confirm two independent sources of truth stay in sync against a
real PostgreSQL database:

1. The tables Alembic actually creates when migrations run (via
   ``alembic upgrade head``) match exactly what the SQLAlchemy ORM
   ``Base.metadata`` declares — catching the class of bug where a migration
   is written but the corresponding ORM model is forgotten, or vice versa.
2. The migration chain is fully applied (database ``alembic_version`` points
   at the expected head revision) and can be safely re-run without error.

Requires a live PostgreSQL instance — see ``tests/conftest.py`` fixtures
(``test_engine``, ``prepared_test_database``) and the ``integration`` pytest
marker applied via ``pytestmark`` below.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.models.base import Base

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_application_tables_exist(test_engine: AsyncEngine) -> None:
    """
    Every table in the live database (excluding Alembic's own bookkeeping
    table) must correspond exactly to a table declared on
    ``Base.metadata``, and vice versa.

    Queries PostgreSQL's ``information_schema`` directly (rather than
    inspecting the ORM) to see the *actual* database state produced by
    running migrations, then compares that set against the ORM's declared
    table names — catching drift in either direction (an orphaned table
    with no ORM model, or an ORM model with no corresponding migration).
    """
    async with test_engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name != 'alembic_version'
                ORDER BY table_name
                """
            ),
        )
        db_tables = {row[0] for row in result.fetchall()}

    assert db_tables == set(Base.metadata.tables.keys())


@pytest.mark.asyncio
async def test_alembic_version_is_head(
    test_engine: AsyncEngine,
    alembic_head_revision: str,
) -> None:
    """
    The database's recorded Alembic revision must match the expected
    ``head`` revision, confirming the full migration chain has been
    applied and nothing is stuck partway through.

    ``alembic_head_revision`` (from ``tests/conftest.py``) must be kept
    manually in sync with the newest file under ``alembic/versions/`` — if
    this test starts failing after adding a new migration, that constant
    needs updating.
    """
    async with test_engine.connect() as connection:
        result = await connection.execute(
            text("SELECT version_num FROM alembic_version"),
        )
        current = result.scalar_one()

    assert current == alembic_head_revision


@pytest.mark.asyncio
async def test_fresh_database_migrates_to_head(test_database_url: str) -> None:
    """
    Re-running ``alembic upgrade head`` against an already-migrated
    database must succeed and leave the schema usable.

    Alembic tracks the applied revision in the ``alembic_version`` table and
    should no-op cleanly when already at head; this guards against a
    migration accidentally being written in a way that isn't idempotent
    (e.g. failing if a table/index/constraint already exists). It queries
    the ``providers`` table afterward as a smoke check that the schema is
    still queryable, not just that the command exited successfully.
    """
    from tests.conftest import run_alembic_upgrade

    run_alembic_upgrade(test_database_url)

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(test_database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT COUNT(*) FROM providers"))
        assert result.scalar_one() >= 0
    await engine.dispose()
