"""
Persistence queries for :class:`~src.models.provider_configuration.ProviderConfiguration`.

A ``ProviderConfiguration`` row holds the one-to-one connection settings
(endpoint, timeouts, TLS/proxy options, provider-specific ``extra_config``)
for a single :class:`~src.models.provider.Provider`. Beyond generic CRUD
from :class:`~src.repositories.base.BaseRepository`, this module adds the
lookup needed to resolve "the" configuration for a given provider and a
convenience listing of active configurations.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.provider_configuration import ProviderConfiguration
from src.repositories.base import BaseRepository


class ProviderConfigurationRepository(BaseRepository[ProviderConfiguration]):
    """
    Data access for provider connection settings.

    Extends :class:`~src.repositories.base.BaseRepository` with a
    provider-scoped lookup (``get_by_provider_id``) that relies on the
    database's one-row-per-provider uniqueness constraint
    (``uq_provider_configurations_provider_id``), plus an ``is_active``
    listing helper.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to ``ProviderConfiguration``.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = ProviderConfigurationRepository(session)
        """
        super().__init__(session, ProviderConfiguration)

    async def get_by_provider_id(self, provider_id: int) -> ProviderConfiguration | None:
        """
        Fetch the 1:1 configuration row for a provider.

        Safe to treat as a unique lookup (rather than "first of many")
        because ``uq_provider_configurations_provider_id`` guarantees at
        most one configuration row exists per provider.

        Args:
            provider_id: Primary key of the owning
                :class:`~src.models.provider.Provider`.

        Returns:
            The provider's ``ProviderConfiguration``, or ``None`` if that
            provider has not been configured yet.

        Raises:
            sqlalchemy.exc.IntegrityError: Propagated from a later
                ``create``/``update`` call (not from this read method) if a
                caller attempts to insert a second configuration row for the
                same ``provider_id``.

        Example:
            >>> config = await repo.get_by_provider_id(1)
            >>> config is None or config.provider_id == 1
            True
        """
        return await self.get_one(ProviderConfiguration.provider_id == provider_id)

    async def list_active(
        self,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ProviderConfiguration]:
        """
        Return active provider configurations.

        Args:
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``ProviderConfiguration`` rows where ``is_active`` is
            ``True``. No explicit ordering is applied (database default,
            typically insertion order), unlike most other ``list_active``
            helpers in this package. Empty list if none are active.

        Example:
            >>> configs = await repo.list_active(limit=50)
            >>> all(c.is_active for c in configs)
            True
        """
        return await self.list(
            ProviderConfiguration.is_active.is_(True),
            offset=offset,
            limit=limit,
        )
