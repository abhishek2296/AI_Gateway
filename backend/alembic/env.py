"""
Alembic migration environment — async engine, shared ORM metadata.

This module is executed by the Alembic CLI (``alembic upgrade``,
``alembic revision --autogenerate``, etc.) every time it runs, in both this
project's supported modes:

- **Online mode** (the normal case): connects to the real database using
  the project's async SQLAlchemy engine/driver (``asyncpg``) and applies
  migrations directly.
- **Offline mode**: emits the raw SQL a migration *would* run to stdout,
  without ever opening a database connection — useful for generating a SQL
  script to hand off to a DBA or run through a separate deployment tool.

``target_metadata`` is set to the same ``Base.metadata`` the application
uses for its ORM models, which is what makes ``alembic revision
--autogenerate`` able to diff the live database schema against the ORM
model definitions and propose a migration automatically.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.core.config import settings
from src.models.base import Base
import src.models  # noqa: F401 — register all ORM tables on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override whatever URL is in alembic.ini with the application's configured
# DATABASE_URL, so a single source of truth (Settings/.env) drives both the
# running app and its migrations.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """
    Configure Alembic to emit SQL to stdout instead of executing it against
    a live database connection.

    Used for ``alembic upgrade head --sql``-style invocations. ``compare_type``
    and ``compare_server_default`` are enabled so autogeneration also
    detects column type/default changes, not just added/removed
    tables/columns.

    Returns:
        None. Runs the configured migrations via
        ``context.run_migrations()`` as a side effect (writing SQL to
        stdout rather than a database).
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Configure Alembic's migration context against an already-open
    synchronous connection and run pending migrations within a transaction.

    This is invoked via ``AsyncConnection.run_sync`` from
    ``run_async_migrations`` below, because Alembic's core migration engine
    is synchronous — it doesn't understand SQLAlchemy's async connection
    API directly, so the async connection must hand off to a sync-style
    callback.

    Args:
        connection: A synchronous ``Connection`` view of the underlying
            async connection, provided automatically by
            ``AsyncConnection.run_sync``.

    Returns:
        None. Applies pending migrations to the database as a side effect.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async SQLAlchemy engine and run migrations against it.

    ``NullPool`` is used because this engine exists only for the lifetime of
    a single Alembic CLI invocation — connection pooling would provide no
    benefit and would leave idle connections open unnecessarily after the
    process exits.

    Returns:
        None. Applies pending migrations to the database as a side effect,
        then disposes of the temporary engine.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Entry point for "online" migrations — i.e. actually connecting to and
    modifying a live database, using the ``asyncpg`` driver.

    Wraps ``run_async_migrations`` in ``asyncio.run`` because Alembic's CLI
    driver code that calls this function is itself synchronous.

    Returns:
        None. Applies pending migrations to the database as a side effect.
    """
    asyncio.run(run_async_migrations())


# Alembic sets this flag based on how it was invoked (e.g. `--sql` implies
# offline mode); dispatch to the appropriate code path accordingly.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
