"""
Helpers shared by persistence integration tests.

Currently holds a single utility, ``expect_integrity_error``, for the common
pattern of asserting that a database operation violates a constraint
(unique key, foreign key, check constraint, etc.) and then recovering the
session so the test (or the ``db_session`` fixture's teardown) can continue
using it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


async def expect_integrity_error(
    session: AsyncSession,
    action: Callable[[], Awaitable[object]],
) -> None:
    """
    Assert that an async action raises ``IntegrityError`` (e.g. from a
    unique, foreign key, or check constraint violation), then roll back the
    session so it remains usable afterward.

    PostgreSQL aborts the current transaction as soon as *any* statement
    fails with an integrity violation — every subsequent statement on that
    same connection/session will also fail until a rollback occurs. Without
    the ``session.rollback()`` here, a test would have to remember to do
    this manually every time it wants to assert a constraint violation and
    then keep using the session (e.g. to verify no row was actually
    inserted).

    Args:
        session: The session the failing ``action`` runs against. Rolled
            back after the expected error is caught.
        action: A zero-argument async callable that performs the operation
            expected to violate a constraint, e.g.
            ``lambda: repo.create(duplicate_entity)``.

    Returns:
        None. Raises an ``AssertionError`` via ``pytest.raises`` if
        ``action`` does *not* raise ``IntegrityError``.

    Raises:
        AssertionError: If ``action()`` completes without raising
            ``IntegrityError`` (surfaced by ``pytest.raises`` itself).

    Example:
        >>> await expect_integrity_error(
        ...     db_session,
        ...     lambda: repo.create(make_provider(name="duplicate-name")),
        ... )  # doctest: +SKIP
    """
    with pytest.raises(IntegrityError):
        await action()
    await session.rollback()
