"""
Persistence queries for :class:`~src.models.ai_model_configuration.AIModelConfiguration`.

An ``AIModelConfiguration`` row holds the one-to-one default generation
settings (temperature, top_p, max_tokens, system prompt template, etc.) for
a single :class:`~src.models.ai_model.AIModel`. This module adds nothing
beyond a single provider-scoped lookup on top of the generic CRUD in
:class:`~src.repositories.base.BaseRepository`, since request-time overrides
of these defaults are handled by the API/service layer, not by this
repository.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model_configuration import AIModelConfiguration
from src.repositories.base import BaseRepository


class AIModelConfigurationRepository(BaseRepository[AIModelConfiguration]):
    """
    Data access for per-model generation defaults.

    Extends :class:`~src.repositories.base.BaseRepository` with a single
    entity-specific lookup, ``get_by_ai_model_id``, which relies on the
    database's one-row-per-model uniqueness constraint
    (``uq_ai_model_configurations_ai_model_id``).
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to ``AIModelConfiguration``.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = AIModelConfigurationRepository(session)
        """
        super().__init__(session, AIModelConfiguration)

    async def get_by_ai_model_id(self, ai_model_id: int) -> AIModelConfiguration | None:
        """
        Fetch the 1:1 configuration row for an AI model.

        Safe to treat as a unique lookup (rather than "first of many")
        because ``uq_ai_model_configurations_ai_model_id`` guarantees at
        most one configuration row exists per model.

        Args:
            ai_model_id: Primary key of the owning
                :class:`~src.models.ai_model.AIModel`.

        Returns:
            The model's ``AIModelConfiguration``, or ``None`` if that model
            has no configured defaults yet (callers should fall back to
            hardcoded/service-layer defaults in that case).

        Raises:
            sqlalchemy.exc.IntegrityError: Propagated from a later
                ``create``/``update`` call (not from this read method) if a
                caller attempts to insert a second configuration row for the
                same ``ai_model_id``.

        Example:
            >>> config = await repo.get_by_ai_model_id(1)
            >>> config is None or config.ai_model_id == 1
            True
        """
        return await self.get_one(AIModelConfiguration.ai_model_id == ai_model_id)
