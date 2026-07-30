"""
Persistence queries for :class:`~src.models.usage_record.UsageRecord`.

A ``UsageRecord`` row is an immutable audit entry for one gateway request
(token counts, latency, estimated cost) used for analytics and billing.
Beyond generic CRUD from :class:`~src.repositories.base.BaseRepository`,
this module adds lookup by the gateway's correlation id (``request_id``)
and two reporting-style range queries: usage within a date window
(optionally scoped to a provider) and usage for a specific provider
(optionally date-bounded).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.usage_record import UsageRecord
from src.repositories.base import BaseRepository


class UsageRecordRepository(BaseRepository[UsageRecord]):
    """
    Data access for request usage and billing audit rows.

    Extends :class:`~src.repositories.base.BaseRepository` with a
    unique-key lookup (``get_by_request_id``) and two reporting queries
    (``usage_between_dates``, ``usage_by_provider``) that build their
    filter list dynamically based on which optional bounds are supplied.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to the ``UsageRecord`` model.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = UsageRecordRepository(session)
        """
        super().__init__(session, UsageRecord)

    async def get_by_request_id(self, request_id: str) -> UsageRecord | None:
        """
        Fetch a usage row by gateway request id.

        ``request_id`` correlates with the HTTP ``X-Request-ID`` middleware
        value and is unique at the database level
        (``uq_usage_records_request_id``) specifically to prevent duplicate
        billing rows if a request is retried — this lookup lets callers
        check "has this request already been recorded?" before inserting.

        Args:
            request_id: The gateway's per-request correlation id, e.g. the
                ``X-Request-ID`` header value.

        Returns:
            The matching ``UsageRecord``, or ``None`` if no usage row has
            been recorded for that request id.

        Example:
            >>> record = await repo.get_by_request_id("req-abc123")
            >>> record is None or record.request_id == "req-abc123"
            True
        """
        return await self.get_one(UsageRecord.request_id == request_id)

    async def usage_between_dates(
        self,
        start: datetime,
        end: datetime,
        *,
        provider_id: int | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[UsageRecord]:
        """
        Return usage rows within a timestamp range.

        Both bounds are inclusive (``>=`` / ``<=``) so callers can pass the
        same instant as both ``start`` and ``end`` to match records at
        exactly that timestamp, and so that day-boundary ranges (e.g.
        midnight-to-midnight) do not silently exclude records that occur
        exactly on the boundary.

        Args:
            start: Inclusive lower bound on ``request_timestamp``.
            end: Inclusive upper bound on ``request_timestamp``.
            provider_id: If given, restrict results to usage rows for this
                provider. Keyword-only. ``None`` (the default) returns
                usage across every provider.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``UsageRecord`` rows within ``[start, end]``, newest
            first (``request_timestamp`` descending). Empty list if none
            match.

        Example:
            >>> from datetime import datetime, timezone
            >>> rows = await repo.usage_between_dates(
            ...     datetime(2026, 1, 1, tzinfo=timezone.utc),
            ...     datetime(2026, 1, 31, tzinfo=timezone.utc),
            ...     provider_id=1,
            ... )
            >>> all(r.provider_id == 1 for r in rows)
            True
        """
        filters = [
            UsageRecord.request_timestamp >= start,
            UsageRecord.request_timestamp <= end,
        ]
        if provider_id is not None:
            filters.append(UsageRecord.provider_id == provider_id)
        return await self.list(
            *filters,
            order_by=(UsageRecord.request_timestamp.desc(),),
            offset=offset,
            limit=limit,
        )

    async def usage_by_provider(
        self,
        provider_id: int,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[UsageRecord]:
        """
        Return usage rows for one provider, optionally date-bounded.

        Complements ``usage_between_dates`` for the common case where the
        caller already knows the provider and wants an optional (rather
        than mandatory) date range — e.g. "all usage for provider X" versus
        "usage for provider X in January".

        Args:
            provider_id: Primary key of the
                :class:`~src.models.provider.Provider` to report usage for.
            start: If given, an inclusive lower bound on
                ``request_timestamp``. Keyword-only. ``None`` means no
                lower bound.
            end: If given, an inclusive upper bound on
                ``request_timestamp``. Keyword-only. ``None`` means no
                upper bound.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``UsageRecord`` rows for that provider (within
            ``[start, end]`` if given), newest first
            (``request_timestamp`` descending). Empty list if none match.

        Example:
            >>> rows = await repo.usage_by_provider(1, limit=100)
            >>> all(r.provider_id == 1 for r in rows)
            True
        """
        filters = [UsageRecord.provider_id == provider_id]
        if start is not None:
            filters.append(UsageRecord.request_timestamp >= start)
        if end is not None:
            filters.append(UsageRecord.request_timestamp <= end)
        return await self.list(
            *filters,
            order_by=(UsageRecord.request_timestamp.desc(),),
            offset=offset,
            limit=limit,
        )
