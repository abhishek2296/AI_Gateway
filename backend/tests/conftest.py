"""
Pytest fixtures for PostgreSQL-backed persistence integration tests.

Tests marked ``@pytest.mark.integration`` (see ``tests/integration/``) need a
real PostgreSQL database rather than mocks, because they verify behavior
that only a real database engine enforces or exhibits — foreign key
cascades, unique/partial-unique constraints, JSONB round-tripping, numeric
precision, and Alembic migrations actually applying cleanly.

This module wires up that database once per test session (creating it and
running Alembic migrations if needed — see ``prepared_test_database``), then
gives each individual test an isolated transactional session
(``db_session``) that is rolled back afterward so tests never see each
other's data, without paying the cost of recreating the schema per test.
"""

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

# Must match the `revision` id of the newest file under alembic/versions/ —
# used by test_alembic.py to assert the test database is fully migrated.
ALEMBIC_HEAD = "c8f5e2a31d04"
BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))

# Snapshot of every ORM-mapped table name, used to TRUNCATE all application
# data between tests that use a *committed* (non-rollback) session.
APPLICATION_TABLES = sorted(Base.metadata.tables.keys())


def resolve_test_database_url() -> str:
    """
    Determine which PostgreSQL database URL integration tests should target.

    Resolution order (first match wins), so CI/local overrides are always
    respected before falling back to a sensible default:

    1. The ``TEST_DATABASE_URL`` environment variable, if set.
    2. ``Settings.TEST_DATABASE_URL``, if configured in ``.env``.
    3. The application's ``DATABASE_URL`` with ``_test`` appended to the
       database path — e.g. ``.../ai_coding_assistant`` becomes
       ``.../ai_coding_assistant_test`` — so integration tests never touch
       the same database as local development by accident.

    Returns:
        A complete PostgreSQL connection URL string dedicated to test use.

    Example:
        >>> import os
        >>> os.environ["TEST_DATABASE_URL"] = "postgresql+asyncpg://u:p@host/db_test"
        >>> resolve_test_database_url()
        'postgresql+asyncpg://u:p@host/db_test'
    """
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
    """
    Derive a connection URL pointing at PostgreSQL's built-in ``postgres``
    maintenance database, given a URL for some other (possibly
    not-yet-existing) database.

    ``CREATE DATABASE`` cannot run while connected to the database being
    created, so ``ensure_test_database_exists`` needs a connection to a
    database that is guaranteed to already exist — the ``postgres``
    maintenance database that ships with every PostgreSQL install.

    Args:
        database_url: The target test database's connection URL, e.g.
            ``postgresql+asyncpg://user:pass@host/ai_coding_assistant_test``.

    Returns:
        The same URL with its path replaced by ``/postgres``.

    Example:
        >>> admin_database_url("postgresql+asyncpg://u:p@host/mydb")
        'postgresql+asyncpg://u:p@host/postgres'
    """
    parsed = urlparse(database_url)
    return urlunparse(parsed._replace(path="/postgres"))


async def ensure_test_database_exists(database_url: str) -> None:
    """
    Create the integration-test database if it doesn't already exist.

    Runs once at the start of a test session so contributors and CI don't
    need to manually provision the test database beforehand.

    Args:
        database_url: Connection URL for the *target* test database (not
            the maintenance database — this function derives that itself
            via ``admin_database_url``).

    Returns:
        None. The database is created as a side effect if missing; if it
        already exists, this is a no-op.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If connecting to the maintenance
            database or executing ``CREATE DATABASE`` fails (e.g. the
            PostgreSQL server is unreachable or credentials are invalid).
    """
    parsed = urlparse(database_url)
    db_name = parsed.path.lstrip("/")
    # AUTOCOMMIT is required because CREATE DATABASE cannot run inside a
    # transaction block in PostgreSQL.
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
    """
    Apply all pending Alembic migrations to the given database via a
    subprocess invocation of the Alembic CLI.

    Runs Alembic as a subprocess (rather than calling its Python API
    in-process) so it uses the exact same code path as running
    ``alembic upgrade head`` manually from a terminal, which is what
    ``test_alembic.py`` is ultimately verifying works correctly.

    Args:
        database_url: The database to migrate. Passed to the subprocess via
            the ``DATABASE_URL`` environment variable, which
            ``alembic/env.py`` reads through ``src.core.config.settings``.

    Returns:
        None. Migrations are applied as a side effect.

    Raises:
        subprocess.CalledProcessError: If the ``alembic upgrade head``
            command exits non-zero (e.g. a migration fails to apply).
    """
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env=env,
        check=True,
    )
    # get_settings() is cached (functools.lru_cache); clear it so any code
    # that reads settings later in the same process picks up the
    # test-specific DATABASE_URL rather than a stale cached value.
    get_settings.cache_clear()


async def truncate_application_tables(session: AsyncSession) -> None:
    """
    Delete all rows from every application table and reset identity
    sequences back to their starting values.

    Used by fixtures/tests that use a *committing* session (as opposed to
    ``db_session``'s rollback-based isolation) — since those changes are
    actually persisted, they must be explicitly cleaned up afterward so
    later tests don't see leftover rows or ID collisions.

    Args:
        session: An active ``AsyncSession`` used to issue the ``TRUNCATE``
            statement and commit it.

    Returns:
        None. All rows in ``APPLICATION_TABLES`` are removed as a side
        effect, and the session's transaction is committed.

    Example:
        >>> await truncate_application_tables(session)  # doctest: +SKIP
    """
    table_list = ", ".join(APPLICATION_TABLES)
    # RESTART IDENTITY resets auto-increment PKs to 1 so subsequent tests
    # get predictable IDs; CASCADE is required because foreign keys link
    # every application table together.
    await session.execute(
        text(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"),
    )
    await session.commit()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """
    Session-scoped fixture exposing the resolved test database URL.

    Session-scoped because the URL is a pure resolution of environment/
    settings values — it cannot change mid-run, so there's no benefit to
    recomputing it per test.

    Returns:
        The connection URL from ``resolve_test_database_url()``.
    """
    return resolve_test_database_url()


@pytest.fixture(scope="session")
def prepared_test_database(test_database_url: str) -> str:
    """
    Ensure the test database exists and is fully migrated, once per test
    session.

    Session-scoped so the (relatively expensive) "create DB + run all
    Alembic migrations" work happens exactly once per test run, not once
    per test — individual test isolation is instead provided by
    ``db_session``'s per-test transaction rollback.

    Args:
        test_database_url: The target database URL (injected by pytest via
            the ``test_database_url`` fixture).

    Returns:
        The same database URL, now guaranteed to exist and be migrated to
        head, passed through for convenience so dependent fixtures don't
        need to depend on both fixtures separately.
    """
    asyncio.run(ensure_test_database_exists(test_database_url))
    run_alembic_upgrade(test_database_url)
    return test_database_url


@pytest_asyncio.fixture(scope="session")
async def test_engine(prepared_test_database: str) -> AsyncGenerator[AsyncEngine, None]:
    """
    Session-scoped async SQLAlchemy engine bound to the prepared test
    database.

    Session-scoped to avoid the overhead of creating a new connection pool
    per test; ``NullPool`` is used because pooling isn't beneficial for a
    short-lived test-session engine and avoids connections lingering
    between test modules.

    Args:
        prepared_test_database: The migrated test database URL (ensures
            this fixture only runs after the database is ready).

    Yields:
        A live ``AsyncEngine`` for the test database. Disposed of
        automatically after the test session finishes.
    """
    engine = create_async_engine(prepared_test_database, poolclass=NullPool, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """
    Function-scoped session wrapped in a connection-level transaction that
    is always rolled back after the test, providing fast, fully isolated
    persistence tests.

    This is the default session fixture most integration tests should use:
    because the outer transaction is rolled back (never committed) at
    teardown, whatever the test creates/modifies/deletes disappears
    automatically — no manual cleanup, no ``TRUNCATE``, and no risk of one
    test's leftover data affecting another. It's also fast, since rollback
    is far cheaper than re-running migrations or truncating tables.

    Note: `session.commit()` calls *inside* a test using this fixture still
    only commit to the outer connection-level transaction, not to the
    database itself — nothing becomes durably visible until the outer
    transaction commits, which never happens here.

    Args:
        test_engine: The session-scoped engine to check out a connection
            from (injected by pytest).

    Yields:
        An ``AsyncSession`` bound to a single connection with an open
        outer transaction. Always rolled back and the connection closed
        during teardown, even if the test raises.
    """
    connection = await test_engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        await session.close()
        # Guard with is_active: a test that itself calls rollback()/commit()
        # on this session may have already ended the transaction.
        if transaction.is_active:
            await transaction.rollback()
        await connection.close()


@pytest_asyncio.fixture
async def uow(db_session: AsyncSession) -> AsyncUnitOfWork:
    """
    ``AsyncUnitOfWork`` bound to the isolated, rollback-based ``db_session``.

    ``close_session=False`` because ``db_session`` owns the underlying
    connection/session lifecycle (via its own teardown) — the Unit of Work
    must not close a session it doesn't own.

    Args:
        db_session: The per-test rollback-isolated session (injected by
            pytest).

    Returns:
        An ``AsyncUnitOfWork`` ready to use in Unit of Work tests.
    """
    return AsyncUnitOfWork(db_session, close_session=False)


@pytest_asyncio.fixture
async def committed_session(
    test_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Function-scoped session that actually commits changes to the database,
    for tests that need to verify cross-session/cross-transaction
    visibility (e.g. "does a second, independent session see what the first
    committed?").

    Unlike ``db_session``, this cannot rely on rollback for isolation —
    committed data really does persist — so teardown instead truncates all
    application tables (``truncate_application_tables``) to reset state for
    the next test.

    Args:
        test_engine: The session-scoped engine to build a session factory
            from (injected by pytest).

    Yields:
        An ``AsyncSession`` from a fresh ``async_sessionmaker``. All
        application tables are truncated during teardown regardless of
        whether the test itself committed.
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
    """
    The expected Alembic ``head`` revision id, used by
    ``test_alembic_version_is_head`` to confirm the test database is fully
    migrated (not stuck on an older revision).

    Returns:
        The revision id string ``ALEMBIC_HEAD``, which must be kept in sync
        with the newest file under ``alembic/versions/``.
    """
    return ALEMBIC_HEAD
