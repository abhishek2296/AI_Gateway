"""Persistence queries for :class:`~src.models.usage_record.UsageRecord`."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.usage_record import UsageRecord
from src.repositories.base import BaseRepository


class UsageRecordRepository(BaseRepository[UsageRecord]):
    """Data access for request usage and billing audit rows."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, UsageRecord)

    async def get_by_request_id(self, request_id: str) -> UsageRecord | None:
        """Fetch a usage row by gateway request id."""
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
        """Return usage rows within a timestamp range."""
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
        """Return usage rows for one provider, optionally date-bounded."""
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
