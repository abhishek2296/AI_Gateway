"""Persistence queries for :class:`~src.models.chat_session.ChatSession`."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.chat_session import ChatSession
from src.repositories.base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Data access for conversation sessions."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatSession)

    async def get_by_uuid(self, session_uuid: uuid.UUID) -> ChatSession | None:
        """Fetch a session by its external UUID."""
        return await self.get_one(ChatSession.session_uuid == session_uuid)

    async def list_recent_sessions(
        self,
        *,
        include_archived: bool = False,
        offset: int | None = None,
        limit: int | None = None,
        load_messages: bool = False,
    ) -> list[ChatSession]:
        """Return sessions ordered by most recently updated."""
        options = [selectinload(ChatSession.messages)] if load_messages else None
        if include_archived:
            return await self.list(
                order_by=(ChatSession.updated_at.desc(),),
                offset=offset,
                limit=limit,
                options=options,
            )
        return await self.list(
            ChatSession.is_archived.is_(False),
            order_by=(ChatSession.updated_at.desc(),),
            offset=offset,
            limit=limit,
            options=options,
        )

    async def archive_session(self, session: ChatSession) -> ChatSession:
        """Mark a session as archived."""
        return await self.update(session, is_archived=True)
