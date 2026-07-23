"""Persistence queries for :class:`~src.models.message.Message`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.message import Message
from src.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Data access for chat messages."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Message)

    async def list_messages(
        self,
        session_id: int,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Return messages for a session in chronological order."""
        return await self.list(
            Message.session_id == session_id,
            order_by=(Message.created_at.asc(),),
            offset=offset,
            limit=limit,
        )

    async def count_for_session(self, session_id: int) -> int:
        """Count messages belonging to a session."""
        return await self.count(Message.session_id == session_id)
