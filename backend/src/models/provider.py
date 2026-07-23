"""ORM entity for an AI provider (Ollama, OpenAI, Anthropic, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String, Text, false, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model import AIModel
    from src.models.api_key import APIKey
    from src.models.chat_session import ChatSession
    from src.models.provider_configuration import ProviderConfiguration
    from src.models.provider_health import ProviderHealth
    from src.models.usage_record import UsageRecord


class Provider(Base, TimestampMixin):
    """
    Registry row for a single LLM backend.

    **Capability ownership**

    - Flags on this entity describe what the **provider API** supports
      (streaming, embeddings, vision, function calling, audio).
    - Flags on :class:`~src.models.ai_model.AIModel` describe whether
      **that specific model** supports a feature.

    Example: a provider may support vision overall, while only some of its
    models set ``supports_images=True``.
    """

    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Stable machine key, e.g. "ollama", "openai", "anthropic"
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    # Human-readable label, e.g. "OpenAI", "Azure OpenAI"
    display_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Provider family / protocol hint, e.g. "ollama", "openai_compatible"
    provider_type: Mapped[str] = mapped_column(String(100), index=True)

    base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )
    is_local: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    # Provider-level API capabilities
    supports_streaming: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_embeddings: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_function_calling: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_vision: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_audio: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    ai_models: Mapped[list[AIModel]] = relationship(
        "AIModel",
        back_populates="provider",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    configuration: Mapped[ProviderConfiguration | None] = relationship(
        "ProviderConfiguration",
        back_populates="provider",
        lazy="selectin",
        uselist=False,
        cascade="all, delete-orphan",
    )
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession",
        back_populates="provider",
        lazy="selectin",
    )
    api_keys: Mapped[list[APIKey]] = relationship(
        "APIKey",
        back_populates="provider",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    usage_records: Mapped[list[UsageRecord]] = relationship(
        "UsageRecord",
        back_populates="provider",
        lazy="selectin",
    )
    health_checks: Mapped[list[ProviderHealth]] = relationship(
        "ProviderHealth",
        back_populates="provider",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="ProviderHealth.checked_at",
    )

    def __repr__(self) -> str:
        return f"<Provider id={self.id} name={self.name!r}>"
