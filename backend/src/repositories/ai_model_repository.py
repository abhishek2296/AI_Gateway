"""Persistence queries for :class:`~src.models.ai_model.AIModel`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model import AIModel
from src.repositories.base import BaseRepository


class AIModelRepository(BaseRepository[AIModel]):
    """Data access for AI model catalog rows."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AIModel)

    async def get_by_provider_and_name(
        self,
        provider_id: int,
        model_name: str,
    ) -> AIModel | None:
        """Fetch a model by provider scope and provider-native model id."""
        return await self.get_one(
            AIModel.provider_id == provider_id,
            AIModel.model_name == model_name,
        )

    async def list_enabled_models(
        self,
        provider_id: int | None = None,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[AIModel]:
        """Return active models, optionally scoped to one provider."""
        filters = [AIModel.is_active.is_(True)]
        if provider_id is not None:
            filters.append(AIModel.provider_id == provider_id)
        return await self.list(
            *filters,
            order_by=(AIModel.model_name.asc(),),
            offset=offset,
            limit=limit,
        )

    async def get_default_model(self, provider_id: int) -> AIModel | None:
        """Return the default active model for a provider, if configured."""
        return await self.get_one(
            AIModel.provider_id == provider_id,
            AIModel.is_default.is_(True),
            AIModel.is_active.is_(True),
        )
