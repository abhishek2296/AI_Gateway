"""
Persistence queries for :class:`~src.models.message.Message`.

A ``Message`` row is one turn (system/user/assistant/tool) within a
:class:`~src.models.chat_session.ChatSession`. Beyond generic CRUD from
:class:`~src.repositories.base.BaseRepository`, this module adds the two
queries a conversation view needs: fetching a session's messages in
chronological order, and counting them (e.g. for context-window bookkeeping
or pagination totals) without loading every row into memory.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.message import Message
from src.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """
    Data access for chat messages.

    Extends :class:`~src.repositories.base.BaseRepository` with two
    session-scoped queries: ``list_messages`` (paginated, chronological) and
    ``count_for_session`` (a lightweight ``SELECT count(*)``).
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to the ``Message`` model.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = MessageRepository(session)
        """
        super().__init__(session, Message)

    async def list_messages(
        self,
        session_id: int,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        Return messages for a session in chronological order.

        Ordered ascending by ``created_at`` (oldest first) so the result can
        be rendered directly as a conversation transcript, matching how a
        chat UI or an LLM prompt-building step would want the messages
        presented.

        Args:
            session_id: Primary key of the owning
                :class:`~src.models.chat_session.ChatSession`.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit (return the full transcript).

        Returns:
            A list of ``Message`` rows for that session, oldest first.
            Empty list if the session has no messages.

        Example:
            >>> messages = await repo.list_messages(1, limit=50)
            >>> messages == sorted(messages, key=lambda m: m.created_at)
            True
        """
        return await self.list(
            Message.session_id == session_id,
            order_by=(Message.created_at.asc(),),
            offset=offset,
            limit=limit,
        )

    async def count_for_session(self, session_id: int) -> int:
        """
        Count messages belonging to a session.

        Uses ``SELECT count(*)`` (inherited ``count`` helper) rather than
        loading every message and taking ``len()``, so this stays cheap even
        for long-running conversations with thousands of messages.

        Args:
            session_id: Primary key of the owning
                :class:`~src.models.chat_session.ChatSession`.

        Returns:
            The number of messages in that session as a plain ``int``.
            ``0`` if the session has no messages (or does not exist).

        Example:
            >>> await repo.count_for_session(1)
            12
        """
        return await self.count(Message.session_id == session_id)
