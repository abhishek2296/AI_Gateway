"""ORM entity for per-request usage analytics and billing."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model import AIModel
    from src.models.chat_session import ChatSession
    from src.models.provider import Provider


class UsageRecord(Base, TimestampMixin):
    """
    Immutable audit row for one gateway AI request.

    Tracks token counts, latency, and estimated cost for analytics and billing.
    ``request_id`` correlates with the HTTP ``X-Request-ID`` middleware value and
    is unique to prevent duplicate billing rows on retries.
    ``chat_session_id`` is optional for stateless or non-session requests.
    """

    __tablename__ = "usage_records"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_usage_records_request_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    chat_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="RESTRICT"),
        index=True,
    )
    ai_model_id: Mapped[int] = mapped_column(
        ForeignKey("ai_models.id", ondelete="RESTRICT"),
        index=True,
    )

    request_id: Mapped[str] = mapped_column(String(64), index=True)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    estimated_cost: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 6),
        nullable=True,
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(50), index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    request_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )

    chat_session: Mapped[ChatSession | None] = relationship(
        "ChatSession",
        back_populates="usage_records",
        lazy="selectin",
    )
    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="usage_records",
        lazy="selectin",
    )
    ai_model: Mapped[AIModel] = relationship(
        "AIModel",
        back_populates="usage_records",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<UsageRecord id={self.id} request_id={self.request_id!r} "
            f"status={self.status!r}>"
        )
