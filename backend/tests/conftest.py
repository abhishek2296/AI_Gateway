"""Pytest fixtures for PostgreSQL-backed persistence integration tests."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncGenerator
from urllib.parse import urlparse, urlunparse

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from src.core.config import get_settings
from src.models.base import Base
from src.unit_of_work import AsyncUnitOfWork

ALEMBIC_HEAD = "c8f5e2a31d04"
BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))

APPLICATION_TABLES = sorted(Base.metadata.tables.keys())


def resolve_test_database_url() -> str:
    """Return the PostgreSQL URL used exclusively for integration tests."""
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit

    settings = get_settings()
    if settings.TEST_DATABASE_URL:
        return settings.TEST_DATABASE_URL

    parsed = urlparse(settings.DATABASE_URL)
    test_path = f"{parsed.path}_test" if parsed.path else "/ai_coding_assistant_test"
    return urlunparse(parsed._replace(path=test_path))


def admin_database_url(database_url: str) -> str:
    """Connect to the maintenance database for CREATE DATABASE operations."""
    parsed = urlparse(database_url)
    return urlunparse(parsed._replace(path="/postgres"))


async def ensure_test_database_exists(database_url: str) -> None:
    """Create the test database when it does not already exist."""
    parsed = urlparse(database_url)
    db_name = parsed.path.lstrip("/")
    engine = create_async_engine(
        admin_database_url(database_url),
        poolclass=NullPool,
        isolation_level="AUTOCOMMIT",
    )

    async with engine.connect() as connection:
        exists = await connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": db_name},
        )
        if exists.scalar_one_or_none() is None:
            await connection.execute(text(f'CREATE DATABASE "{db_name}"'))

    await engine.dispose()


def run_alembic_upgrade(database_url: str) -> None:
    """Apply all migrations to the test database via subprocess."""
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env=env,
        check=True,
    )
    get_settings.cache_clear()


async def truncate_application_tables(session: AsyncSession) -> None:
    """Remove all rows from application tables and reset identity sequences."""
    table_list = ", ".join(APPLICATION_TABLES)
    await session.execute(
        text(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"),
    )
    await session.commit()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    return resolve_test_database_url()


@pytest.fixture(scope="session")
def prepared_test_database(test_database_url: str) -> str:
    """Create the test database and apply Alembic migrations once per session."""
    asyncio.run(ensure_test_database_exists(test_database_url))
    run_alembic_upgrade(test_database_url)
    return test_database_url


@pytest_asyncio.fixture(scope="session")
async def test_engine(prepared_test_database: str) -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(prepared_test_database, poolclass=NullPool, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a session wrapped in a connection-level transaction that rolls back
    after each test for fast, isolated persistence checks.
    """
    connection = await test_engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        await session.close()
        if transaction.is_active:
            await transaction.rollback()
        await connection.close()


@pytest_asyncio.fixture
async def uow(db_session: AsyncSession) -> AsyncUnitOfWork:
    """Unit of Work bound to the isolated test session."""
    return AsyncUnitOfWork(db_session, close_session=False)


@pytest_asyncio.fixture
async def committed_session(
    test_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Session that commits changes — truncates all tables after each test.
    Used for Unit of Work commit and cross-session visibility tests.
    """
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        try:
            yield session
        finally:
            await truncate_application_tables(session)


@pytest.fixture
def alembic_head_revision() -> str:
    return ALEMBIC_HEAD
