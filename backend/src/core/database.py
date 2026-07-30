"""
Async SQLAlchemy engine and session-factory configuration.

This module owns the single, process-wide database engine and session
factory for the AI Gateway's persistence layer. It is intentionally the only
place ``create_async_engine``/``async_sessionmaker`` are called — everything
else (``core/session.py``'s ``get_session`` dependency, ``core/lifespan.py``'s
shutdown hook) builds on top of the ``engine`` and ``AsyncSessionLocal``
objects defined here, so there is exactly one connection pool per process.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import settings


# Module-level singleton: created once at import time using the process's
# configured settings, then reused by every session for the life of the
# application. `pool_pre_ping=True` makes the pool issue a lightweight
# "is this connection still alive?" check before handing out a pooled
# connection, which avoids surfacing stale-connection errors (e.g. after the
# database restarts or an idle connection is dropped by a firewall/proxy) to
# application code.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

# `expire_on_commit=False` keeps ORM objects' attributes readable after a
# `commit()` without triggering an implicit re-fetch from the database on
# next access — important for async sessions, where that implicit refresh
# would otherwise require an extra `await` the caller isn't expecting.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def close_database_connection() -> None:
    """
    Release every pooled connection during application shutdown.

    Calls ``engine.dispose()``, which closes all connections currently held
    in the pool and discards the pool itself. This should be called exactly
    once, during the application's shutdown sequence (see
    ``core/lifespan.py``), so the process does not leave open database
    connections behind when it exits.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If disposing the underlying
            connection pool fails unexpectedly (rare in practice, since
            ``dispose()`` is designed to be safe to call even on an already
            idle pool).

    Example:
        >>> import asyncio
        >>> asyncio.run(close_database_connection())  # doctest: +SKIP
    """
    await engine.dispose()
