"""ORM entity for an AI model offered by a provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint, false, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model_configuration import AIModelConfiguration
    from src.models.chat_session import ChatSession
    from src.models.provider import Provider
    from src.models.usage_record import UsageRecord


class AIModel(Base, TimestampMixin):
    """
    A concrete model identifier under a :class:`~src.models.provider.Provider`.

    Capability flags here mean: **this specific model** supports the feature.
    Provider-level API capabilities live on :class:`~src.models.provider.Provider`.

    ``model_name`` is the provider-native id (e.g. ``qwen3:8b``, ``gpt-4o``).
    Uniqueness is scoped per provider so different backends may share names.
    """

    __tablename__ = "ai_models"
    __table_args__ = (
        UniqueConstraint(
            "provider_id",
            "model_name",
            name="uq_ai_models_provider_id_model_name",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    model_name: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Model-level capabilities (this model only)
    supports_tools: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_json: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_images: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_streaming: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="ai_models",
        lazy="selectin",
    )
    configuration: Mapped[AIModelConfiguration | None] = relationship(
        "AIModelConfiguration",
        back_populates="ai_model",
        lazy="selectin",
        uselist=False,
        cascade="all, delete-orphan",
    )
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession",
        back_populates="ai_model",
        lazy="selectin",
    )
    usage_records: Mapped[list[UsageRecord]] = relationship(
        "UsageRecord",
        back_populates="ai_model",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<AIModel id={self.id} model_name={self.model_name!r} "
            f"provider_id={self.provider_id}>"
        )
