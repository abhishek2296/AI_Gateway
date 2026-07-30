"""
Persistence queries for :class:`~src.models.ai_model.AIModel`.

An ``AIModel`` row is a concrete model identifier under a
:class:`~src.models.provider.Provider` (e.g. ``qwen3:8b`` under the
``ollama`` provider). Beyond generic CRUD from
:class:`~src.repositories.base.BaseRepository`, this module adds the
provider-scoped lookups needed to resolve "which model" a request should
use: by provider + native name, by "enabled" status, and by the
provider's configured default model.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model import AIModel
from src.repositories.base import BaseRepository


class AIModelRepository(BaseRepository[AIModel]):
    """
    Data access for AI model catalog rows.

    Extends :class:`~src.repositories.base.BaseRepository` with queries that
    respect ``AIModel``'s provider-scoped uniqueness and "one default per
    provider" invariants (both enforced at the database level via
    constraints in ``models/ai_model.py``).
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to the ``AIModel`` model.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = AIModelRepository(session)
        """
        super().__init__(session, AIModel)

    async def get_by_provider_and_name(
        self,
        provider_id: int,
        model_name: str,
    ) -> AIModel | None:
        """
        Fetch a model by provider scope and provider-native model id.

        Filters on the composite ``(provider_id, model_name)`` pair because
        ``model_name`` (e.g. ``"gpt-4o"``) is only unique *within* a
        provider (see ``uq_ai_models_provider_id_model_name`` in
        ``models/ai_model.py``) — two different providers may legitimately
        offer models with the same native name.

        Args:
            provider_id: Primary key of the owning
                :class:`~src.models.provider.Provider`.
            model_name: The provider-native model identifier, e.g.
                ``"qwen3:8b"``.

        Returns:
            The matching ``AIModel``, or ``None`` if no model with that name
            is registered under that provider.

        Example:
            >>> model = await repo.get_by_provider_and_name(1, "qwen3:8b")
            >>> model is None or model.model_name == "qwen3:8b"
            True
        """
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
        """
        Return active models, optionally scoped to one provider.

        Args:
            provider_id: If given, restrict results to models owned by this
                provider. ``None`` (the default) returns active models
                across every provider.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``AIModel`` rows where ``is_active`` is ``True``,
            sorted alphabetically by ``model_name``. Empty list if none
            match.

        Example:
            >>> models = await repo.list_enabled_models(provider_id=1)
            >>> all(m.is_active for m in models)
            True
        """
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
        """
        Return the default active model for a provider, if configured.

        A partial unique index (``uq_ai_models_one_default_per_provider``)
        guarantees at most one row per provider can have
        ``is_default=True`` at the database level, so this query can never
        return an ambiguous result even without an explicit ``LIMIT``.

        Args:
            provider_id: Primary key of the
                :class:`~src.models.provider.Provider` to look up the
                default model for.

        Returns:
            The active default ``AIModel`` for that provider, or ``None``
            if the provider has no model marked as default (or the default
            model has since been deactivated).

        Example:
            >>> default_model = await repo.get_default_model(1)
            >>> default_model is None or default_model.is_default
            True
        """
        return await self.get_one(
            AIModel.provider_id == provider_id,
            AIModel.is_default.is_(True),
            AIModel.is_active.is_(True),
        )
