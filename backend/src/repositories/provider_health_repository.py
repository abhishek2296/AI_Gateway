"""Persistence queries for :class:`~src.models.provider_health.ProviderHealth`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.provider_health import ProviderHealth
from src.repositories.base import BaseRepository


class ProviderHealthRepository(BaseRepository[ProviderHealth]):
    """Data access for provider health check snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ProviderHealth)

    async def latest_health(self, provider_id: int) -> ProviderHealth | None:
        """Return the most recent health snapshot for a provider."""
        rows = await self.list(
            ProviderHealth.provider_id == provider_id,
            order_by=(ProviderHealth.checked_at.desc(),),
            limit=1,
        )
        return rows[0] if rows else None

    async def failed_checks(
        self,
        provider_id: int,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ProviderHealth]:
        """Return non-healthy snapshots for a provider, newest first."""
        return await self.list(
            ProviderHealth.provider_id == provider_id,
            ProviderHealth.status != "healthy",
            order_by=(ProviderHealth.checked_at.desc(),),
            offset=offset,
            limit=limit,
        )

    async def list_for_provider(
        self,
        provider_id: int,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ProviderHealth]:
        """Return health history for a provider ordered by check time."""
        return await self.list(
            ProviderHealth.provider_id == provider_id,
            order_by=(ProviderHealth.checked_at.desc(),),
            offset=offset,
            limit=limit,
        )
