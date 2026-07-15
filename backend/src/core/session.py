"""Helpers for creating short-lived async database sessions."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import AsyncSessionLocal


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield one session and ensure it is closed when its caller finishes."""
    async with AsyncSessionLocal() as session:
        yield session

