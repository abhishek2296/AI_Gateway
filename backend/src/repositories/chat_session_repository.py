"""
Persistence queries for :class:`~src.models.chat_session.ChatSession`.

A ``ChatSession`` row represents one conversation thread bound to a
provider and AI model. Beyond generic CRUD from
:class:`~src.repositories.base.BaseRepository`, this module adds lookup by
external UUID (the identifier exposed to API clients, as opposed to the
internal integer primary key), a "recent sessions" feed with optional
archived-session filtering and optional eager-loading of messages, and an
``archive_session`` convenience wrapper.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.chat_session import ChatSession
from src.repositories.base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    """
    Data access for conversation sessions.

    Extends :class:`~src.repositories.base.BaseRepository` with
    session-specific lookups (``get_by_uuid``), a paginated "recent
    sessions" query with optional relationship eager-loading
    (``list_recent_sessions``), and a small semantic wrapper around the
    inherited ``update`` for archiving (``archive_session``).
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to the ``ChatSession`` model.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = ChatSessionRepository(session)
        """
        super().__init__(session, ChatSession)

    async def get_by_uuid(self, session_uuid: uuid.UUID) -> ChatSession | None:
        """
        Fetch a session by its external UUID.

        ``ChatSession.session_uuid`` is the identifier exposed to API
        clients (unique, indexed) — the internal integer ``id`` primary key
        is never leaked outside the persistence layer, so this is the
        lookup used by any route/service handling a client-supplied session
        reference.

        Args:
            session_uuid: The externally-visible session identifier, e.g.
                parsed from a URL path parameter.

        Returns:
            The matching ``ChatSession``, or ``None`` if no session has
            that UUID.

        Example:
            >>> session = await repo.get_by_uuid(uuid.UUID("..."))
            >>> session is None or session.session_uuid == some_uuid
            True
        """
        return await self.get_one(ChatSession.session_uuid == session_uuid)

    async def list_recent_sessions(
        self,
        *,
        include_archived: bool = False,
        offset: int | None = None,
        limit: int | None = None,
        load_messages: bool = False,
    ) -> list[ChatSession]:
        """
        Return sessions ordered by most recently updated.

        ``updated_at`` (rather than ``created_at``) drives the ordering so
        that sessions with recent activity (new messages, edited settings)
        surface at the top, matching the "recently active conversations"
        UX most chat clients expect.

        When ``load_messages`` is ``True``, a ``selectinload`` option is
        added so each session's ``messages`` relationship is fetched via a
        single extra batched query (``WHERE session_id IN (...)``) instead
        of one query per session — avoiding the N+1 query problem a naive
        lazy-load would cause when the caller then iterates over every
        session's messages (e.g. to render a session list with previews).

        Args:
            include_archived: If ``False`` (the default), only sessions
                where ``is_archived`` is ``False`` are returned. If
                ``True``, both archived and active sessions are returned.
                Keyword-only.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.
            load_messages: If ``True``, eagerly load each session's
                ``messages`` relationship via ``selectinload`` so accessing
                ``session.messages`` afterwards does not trigger additional
                lazy-load queries. Keyword-only, defaults to ``False``.

        Returns:
            A list of ``ChatSession`` rows ordered by ``updated_at``
            descending (most recently updated first). Empty list if none
            match.

        Example:
            >>> sessions = await repo.list_recent_sessions(limit=20, load_messages=True)
            >>> all(not s.is_archived for s in sessions)
            True
        """
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
        """
        Mark a session as archived.

        Thin, semantically-named wrapper around the inherited ``update`` so
        call sites read as an intent (``archive_session(session)``) rather
        than a generic attribute mutation (``update(session, is_archived=True)``).

        Args:
            session: An already-persisted ``ChatSession`` to archive.

        Returns:
            The same ``session`` instance, refreshed, with ``is_archived``
            set to ``True``.

        Example:
            >>> archived = await repo.archive_session(session)
            >>> archived.is_archived
            True
        """
        return await self.update(session, is_archived=True)
