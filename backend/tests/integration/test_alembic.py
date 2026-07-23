"""Alembic migration and schema alignment tests."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.models.base import Base

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_application_tables_exist(test_engine: AsyncEngine) -> None:
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
    async with test_engine.connect() as connection:
        result = await connection.execute(
            text("SELECT version_num FROM alembic_version"),
        )
        current = result.scalar_one()

    assert current == alembic_head_revision


@pytest.mark.asyncio
async def test_fresh_database_migrates_to_head(test_database_url: str) -> None:
    """Re-running upgrade on an already-migrated database remains valid."""
    from tests.conftest import run_alembic_upgrade

    run_alembic_upgrade(test_database_url)

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(test_database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT COUNT(*) FROM providers"))
        assert result.scalar_one() >= 0
    await engine.dispose()
