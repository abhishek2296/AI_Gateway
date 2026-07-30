"""
ORM entity for one message within a chat session.

Each ``Message`` row is a single turn (system prompt, user input, assistant
reply, or tool output) belonging to exactly one
:class:`~src.models.chat_session.ChatSession`. Together, the ordered set of
messages for a session forms the full conversation transcript sent to and
received from the provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.chat_session import ChatSession


class Message(Base, TimestampMixin):
    """
    A single message in a :class:`~src.models.chat_session.ChatSession`.

    Maps to the ``messages`` table. ``role`` stores the message role as a
    string (``system``, ``user``, ``assistant``, ``tool``) rather than a
    database enum, following the string-identity pattern in ADR-007 — this
    keeps the set of valid roles extensible (e.g. future provider-specific
    roles) without requiring a schema migration for every new value.
    Messages are ordered chronologically by ``created_at`` (see
    :attr:`~src.models.chat_session.ChatSession.messages`).

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        session_id: Foreign key to ``chat_sessions.id``.
            ``ondelete="CASCADE"`` — a message cannot exist without its
            session, so deleting a session deletes all of its messages.
            Indexed for fast "all messages in this session" lookups.
        role: Who/what produced this message: ``"system"``, ``"user"``,
            ``"assistant"``, or ``"tool"``. ``String(50)``, indexed.
        content: The message text/body. ``Text``, required — even an empty
            assistant response should be represented as ``""`` rather than
            omitted, since the row's existence marks that a turn occurred.
        extra_metadata: Free-form JSON metadata for this message (e.g.
            tool-call payloads, citations). ``JSONB``, nullable. Mapped
            from the database column literally named ``"metadata"`` to the
            Python attribute ``extra_metadata`` because ``metadata`` is
            reserved on SQLAlchemy declarative classes.
        token_count: Number of tokens this message consumed/produced, or
            ``None`` if not tracked (e.g. provider didn't report usage).
            ``Integer``, nullable.
        latency_ms: Time in milliseconds the provider took to produce this
            message (only meaningful for assistant messages), or ``None``.
            ``Integer``, nullable.
        provider_response_id: The upstream provider's own response/request
            identifier for this message, useful for correlating with
            provider-side logs or support tickets. ``String(255)``,
            nullable, indexed.
        finish_reason: Why the provider stopped generating (e.g.
            ``"stop"``, ``"length"``, ``"tool_calls"``), or ``None`` for
            non-assistant messages or providers that don't report it.
            ``String(100)``, nullable.
        chat_session: Many-to-one back to
            :class:`~src.models.chat_session.ChatSession`. Eagerly loaded
            (``lazy="selectin"``).

    Example:
        >>> message = Message(
        ...     session_id=1,
        ...     role="user",
        ...     content="What's the weather like?",
        ... )
        >>> message.role
        'user'
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )

    role: Mapped[str] = mapped_column(String(50), index=True)
    content: Mapped[str] = mapped_column(Text)

    # Python attribute is "extra_metadata" (not "metadata") because
    # "metadata" is reserved by SQLAlchemy's declarative base; the column
    # itself is still named "metadata" in the actual database table.
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider_response_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    finish_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)

    chat_session: Mapped[ChatSession] = relationship(
        "ChatSession",
        back_populates="messages",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<Message id=1 session_id=1 role='user'>``.
            ``content`` is deliberately omitted since message bodies can be
            long and may contain sensitive user input that shouldn't be
            dumped into logs by default.

        Example:
            >>> Message(id=1, session_id=1, role="user").__repr__()
            "<Message id=1 session_id=1 role='user'>"
        """
        return (
            f"<Message id={self.id} session_id={self.session_id} "
            f"role={self.role!r}>"
        )
