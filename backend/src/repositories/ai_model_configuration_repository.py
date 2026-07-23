"""Persistence queries for :class:`~src.models.ai_model_configuration.AIModelConfiguration`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model_configuration import AIModelConfiguration
from src.repositories.base import BaseRepository


class AIModelConfigurationRepository(BaseRepository[AIModelConfiguration]):
    """Data access for per-model generation defaults."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AIModelConfiguration)

    async def get_by_ai_model_id(self, ai_model_id: int) -> AIModelConfiguration | None:
        """Fetch the 1:1 configuration row for an AI model."""
        return await self.get_one(AIModelConfiguration.ai_model_id == ai_model_id)
