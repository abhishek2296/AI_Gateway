"""Persistence queries for :class:`~src.models.provider.Provider`."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model import AIModel
from src.models.provider import Provider
from src.repositories.base import BaseRepository


class ProviderRepository(BaseRepository[Provider]):
    """Data access for provider catalog rows."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Provider)

    async def get_by_name(self, name: str) -> Provider | None:
        """Fetch a provider by its stable machine key."""
        return await self.get_one(Provider.name == name)

    async def get_default_provider(self) -> Provider | None:
        """
        Return the active provider that owns a default model.

        Falls back to the first active provider ordered by name when no default
        model is configured.
        """
        stmt = (
            select(Provider)
            .join(AIModel, AIModel.provider_id == Provider.id)
            .where(
                Provider.is_active.is_(True),
                AIModel.is_default.is_(True),
                AIModel.is_active.is_(True),
            )
            .order_by(Provider.name.asc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        provider = result.scalar_one_or_none()
        if provider is not None:
            return provider

        active = await self.list_active(limit=1)
        return active[0] if active else None

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
