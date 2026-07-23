"""ORM entity for a conversation bound to a provider and AI model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, Uuid, false
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model import AIModel
    from src.models.message import Message
    from src.models.provider import Provider
    from src.models.usage_record import UsageRecord


class ChatSession(Base, TimestampMixin):
    """
    A single conversation thread with a selected provider and AI model.

    ``provider_id`` is denormalized for fast filtering; ``ai_model_id`` is
    the source of truth for model selection. The service layer must ensure
    ``ai_model.provider_id == provider_id`` when creating or updating sessions.

    Session-level overrides (``system_prompt``, ``temperature_override``,
    ``max_tokens_override``) take precedence over catalog defaults at runtime.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    session_uuid: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        unique=True,
        index=True,
    )

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )
    ai_model_id: Mapped[int] = mapped_column(
        ForeignKey("ai_models.id", ondelete="CASCADE"),
        index=True,
    )

    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    temperature_override: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens_override: Mapped[int | None] = mapped_column(Integer, nullable=True)

    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
        index=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="chat_sessions",
        lazy="selectin",
    )
    ai_model: Mapped[AIModel] = relationship(
        "AIModel",
        back_populates="chat_sessions",
        lazy="selectin",
    )
    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="chat_session",
        lazy="selectin",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )
    usage_records: Mapped[list[UsageRecord]] = relationship(
        "UsageRecord",
        back_populates="chat_session",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<ChatSession id={self.id} session_uuid={self.session_uuid!s} "
            f"provider_id={self.provider_id} ai_model_id={self.ai_model_id}>"
        )
