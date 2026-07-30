"""
Persistence queries for :class:`~src.models.provider.Provider`.

A ``Provider`` row represents one LLM backend registered in the gateway's
catalog (Ollama, OpenAI, Anthropic, ...). Beyond the generic CRUD inherited
from :class:`~src.repositories.base.BaseRepository`, this module adds the
lookups the gateway's routing/service layer needs at request time: finding a
provider by its stable machine key, discovering "the" default provider to
route to when a caller doesn't specify one, and listing active providers.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model import AIModel
from src.models.provider import Provider
from src.repositories.base import BaseRepository


class ProviderRepository(BaseRepository[Provider]):
    """
    Data access for provider catalog rows.

    Extends :class:`~src.repositories.base.BaseRepository` (create,
    ``get_by_id``, ``update``, ``delete``, ``count``, ``exists``, ...) with
    provider-specific lookups: ``get_by_name`` (unique machine key),
    ``get_default_provider`` (routing fallback logic spanning
    :class:`~src.models.ai_model.AIModel`), and ``list_active``.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to the ``Provider`` model.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = ProviderRepository(session)
        """
        super().__init__(session, Provider)

    async def get_by_name(self, name: str) -> Provider | None:
        """
        Fetch a provider by its stable machine key.

        ``Provider.name`` (e.g. ``"openai"``, ``"ollama"``) is unique at the
        database level (see ``providers.name`` in ``models/provider.py``),
        so this is the canonical way to resolve a provider from a
        config/env value or an API request's ``provider`` field.

        Args:
            name: The provider's machine key, e.g. ``"openai"``. Case- and
                whitespace-sensitive — must match the stored value exactly.

        Returns:
            The matching ``Provider``, or ``None`` if no provider is
            registered under that name.

        Example:
            >>> provider = await repo.get_by_name("openai")
            >>> provider is None or provider.name == "openai"
            True
        """
        return await self.get_one(Provider.name == name)

    async def get_default_provider(self) -> Provider | None:
        """
        Return the active provider that owns a default model.

        Falls back to the first active provider ordered by name when no default
        model is configured.

        The primary query joins to :class:`~src.models.ai_model.AIModel`
        because "default" is modeled at the model level
        (``AIModel.is_default``), not on ``Provider`` itself — a provider is
        only considered "the" default if it is active *and* currently has an
        active model flagged as the default. The application enforces at
        most one such model per provider (see the partial unique index
        ``uq_ai_models_one_default_per_provider`` in ``models/ai_model.py``),
        so this join can only ever match zero or one provider; ``.limit(1)``
        combined with the name ordering is defensive rather than strictly
        required, but keeps behavior deterministic if that invariant were
        ever violated.

        If no provider has a default model configured yet (e.g. on a fresh
        install before an admin sets one up), this falls back to
        ``list_active`` so the gateway still has *some* provider to route
        to, rather than returning ``None`` and failing every request.

        Returns:
            The provider that owns the active default model, if one exists;
            otherwise the first active provider ordered by name; otherwise
            ``None`` if there are no active providers at all.

        Example:
            >>> provider = await repo.get_default_provider()
            >>> provider is None or provider.is_active
            True
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
        """
        Return active providers ordered by name.

        Used both directly (e.g. an admin listing endpoint) and internally
        by ``get_default_provider`` as its fallback path.

        Args:
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``Provider`` rows where ``is_active`` is ``True``,
            sorted alphabetically by ``name``. Empty list if none are active.

        Example:
            >>> providers = await repo.list_active(limit=20)
            >>> all(p.is_active for p in providers)
            True
        """
        return await self.list(
            Provider.is_active.is_(True),
            order_by=(Provider.name.asc(),),
            offset=offset,
            limit=limit,
        )
