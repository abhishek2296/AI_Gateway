"""ORM entity for one message within a chat session."""

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

    ``role`` stores the message role as a string (``system``, ``user``,
    ``assistant``, ``tool``) following the string-identity pattern in ADR-007.
    Messages are ordered chronologically by ``created_at``.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )

    role: Mapped[str] = mapped_column(String(50), index=True)
    content: Mapped[str] = mapped_column(Text)

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
        return (
            f"<Message id={self.id} session_id={self.session_id} "
            f"role={self.role!r}>"
        )
