"""Persistence queries for :class:`~src.models.provider_configuration.ProviderConfiguration`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.provider_configuration import ProviderConfiguration
from src.repositories.base import BaseRepository


class ProviderConfigurationRepository(BaseRepository[ProviderConfiguration]):
    """Data access for provider connection settings."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ProviderConfiguration)

    async def get_by_provider_id(self, provider_id: int) -> ProviderConfiguration | None:
        """Fetch the 1:1 configuration row for a provider."""
        return await self.get_one(ProviderConfiguration.provider_id == provider_id)

    async def list_active(
        self,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ProviderConfiguration]:
        """Return active provider configurations."""
        return await self.list(
            ProviderConfiguration.is_active.is_(True),
            offset=offset,
            limit=limit,
        )
