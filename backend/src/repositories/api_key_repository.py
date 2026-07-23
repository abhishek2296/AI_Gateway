"""Persistence queries for :class:`~src.models.api_key.APIKey`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_key import APIKey
from src.repositories.base import BaseRepository


class APIKeyRepository(BaseRepository[APIKey]):
    """Data access for provider credential references."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, APIKey)

    async def get_by_provider_and_name(
        self,
        provider_id: int,
        name: str,
    ) -> APIKey | None:
        """Fetch a credential row by provider and logical name."""
        return await self.get_one(
            APIKey.provider_id == provider_id,
            APIKey.name == name,
        )

    async def get_default_for_provider(self, provider_id: int) -> APIKey | None:
        """Return the default active key for a provider, if configured."""
        return await self.get_one(
            APIKey.provider_id == provider_id,
            APIKey.is_default.is_(True),
            APIKey.is_active.is_(True),
        )

    async def list_active_for_provider(
        self,
        provider_id: int,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[APIKey]:
        """Return active keys for a provider ordered by name."""
        return await self.list(
            APIKey.provider_id == provider_id,
            APIKey.is_active.is_(True),
            order_by=(APIKey.name.asc(),),
            offset=offset,
            limit=limit,
        )
