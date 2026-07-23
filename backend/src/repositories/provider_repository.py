"""Persistence queries for :class:`~src.models.provider.Provider`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.provider import Provider
from src.repositories.base import BaseRepository


class ProviderRepository(BaseRepository[Provider]):
    """Data access for provider catalog rows."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Provider)

    async def get_by_name(self, name: str) -> Provider | None:
        """Fetch a provider by its stable machine key."""
        return await self.get_one(Provider.name == name)

    async def list_active(
        self,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Provider]:
        """Return active providers ordered by name."""
        return await self.list(
            Provider.is_active.is_(True),
            order_by=(Provider.name.asc(),),
            offset=offset,
            limit=limit,
        )
