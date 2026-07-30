"""
Persistence queries for :class:`~src.models.provider_health.ProviderHealth`.

A ``ProviderHealth`` row is a single point-in-time health-check probe
result for a :class:`~src.models.provider.Provider` (append-only; history
is never mutated). Beyond generic CRUD from
:class:`~src.repositories.base.BaseRepository`, this module adds the
queries a health/monitoring dashboard needs: the latest snapshot ("is the
provider healthy *right now*?"), only the non-healthy snapshots (for
incident review), and the full paginated history for one provider.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.provider_health import ProviderHealth
from src.repositories.base import BaseRepository


class ProviderHealthRepository(BaseRepository[ProviderHealth]):
    """
    Data access for provider health check snapshots.

    Extends :class:`~src.repositories.base.BaseRepository` with three
    ``checked_at``-ordered queries scoped to a single provider:
    ``latest_health``, ``failed_checks``, and ``list_for_provider``.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to ``ProviderHealth``.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = ProviderHealthRepository(session)
        """
        super().__init__(session, ProviderHealth)

    async def latest_health(self, provider_id: int) -> ProviderHealth | None:
        """
        Return the most recent health snapshot for a provider.

        Implemented via ``list(..., limit=1)`` rather than ``get_one``
        purely so the single result can be ordered by ``checked_at``
        descending — ``get_one`` does not accept an ``order_by`` parameter,
        since it is meant for filters that are already unique.

        Args:
            provider_id: Primary key of the
                :class:`~src.models.provider.Provider` to check.

        Returns:
            The ``ProviderHealth`` row with the greatest ``checked_at``
            value for that provider, or ``None`` if the provider has never
            been health-checked.

        Example:
            >>> health = await repo.latest_health(1)
            >>> health is None or health.provider_id == 1
            True
        """
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
        """
        Return non-healthy snapshots for a provider, newest first.

        Filters on ``status != "healthy"`` (rather than an explicit
        allow-list of "bad" statuses like ``"unhealthy"``/``"degraded"``)
        so any future status value that isn't ``"healthy"`` is
        automatically treated as noteworthy here without needing to update
        this query.

        Args:
            provider_id: Primary key of the
                :class:`~src.models.provider.Provider` to inspect.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``ProviderHealth`` rows for that provider where
            ``status`` is not ``"healthy"``, newest first (``checked_at``
            descending). Empty list if the provider has no failed checks.

        Example:
            >>> failures = await repo.failed_checks(1, limit=10)
            >>> all(f.status != "healthy" for f in failures)
            True
        """
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
        """
        Return health history for a provider ordered by check time.

        Args:
            provider_id: Primary key of the
                :class:`~src.models.provider.Provider` to inspect.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of every ``ProviderHealth`` row for that provider
            (healthy and non-healthy alike), newest first (``checked_at``
            descending). Empty list if the provider has never been
            health-checked.

        Example:
            >>> history = await repo.list_for_provider(1, limit=50)
            >>> all(h.provider_id == 1 for h in history)
            True
        """
        return await self.list(
            ProviderHealth.provider_id == provider_id,
            order_by=(ProviderHealth.checked_at.desc(),),
            offset=offset,
            limit=limit,
        )
