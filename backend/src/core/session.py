"""
Helpers for creating short-lived async database sessions.

This module provides the FastAPI dependency-injection entry point for
database access: routes and services should depend on ``get_session`` (via
``fastapi.Depends``) rather than importing ``AsyncSessionLocal`` from
``core/database.py`` directly, so session lifetime is always scoped to a
single request and sessions are guaranteed to be closed even if the request
handler raises.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import AsyncSessionLocal


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield one database session and ensure it is closed when its caller finishes.

    Designed to be used as a FastAPI dependency, e.g.
    ``session: AsyncSession = Depends(get_session)``. FastAPI drives this
    generator: it resumes execution up to (and including) the ``yield`` to
    obtain the session, hands that session to the route/service, and then
    resumes the generator after the request completes (successfully or with
    an exception) so the ``async with`` block's ``__aexit__`` runs and the
    session is released back to the connection pool.

    Because a new session is created per call, concurrent requests never
    share a session — avoiding the thread/async-safety issues that would
    arise from reusing one session across requests.

    Yields:
        A fresh ``AsyncSession`` bound to the shared engine configured in
        ``core/database.py``, valid for the duration of one request.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: Propagates if session creation itself
            fails (e.g. the underlying connection pool cannot obtain a
            connection). Errors raised by the caller *while using* the
            session (e.g. a failed query) also propagate through this
            generator, but the session is still closed via the ``async with``
            block before the exception continues upward.

    Example:
        >>> from fastapi import Depends
        >>> async def some_route(session=Depends(get_session)):  # doctest: +SKIP
        ...     ...
    """
    async with AsyncSessionLocal() as session:
        yield session

