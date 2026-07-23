"""Helpers shared by persistence integration tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


async def expect_integrity_error(
    session: AsyncSession,
    action: Callable[[], Awaitable[object]],
) -> None:
    """Assert an action raises IntegrityError and reset the aborted transaction."""
    with pytest.raises(IntegrityError):
        await action()
    await session.rollback()
