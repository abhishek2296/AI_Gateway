"""ORM entity for per-model generation and tool defaults."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint, false
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model import AIModel


class AIModelConfiguration(Base, TimestampMixin):
    """
    One-to-one default generation settings for an
    :class:`~src.models.ai_model.AIModel`.

    These are catalog defaults (temperature, system prompt template, etc.).
    Request-time overrides belong in the API/service layer, not here.
    Provider-specific extras go in ``extra_parameters``.
    """

    __tablename__ = "ai_model_configurations"
    __table_args__ = (
        UniqueConstraint("ai_model_id", name="uq_ai_model_configurations_ai_model_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    ai_model_id: Mapped[int] = mapped_column(
        ForeignKey("ai_models.id", ondelete="CASCADE"),
        index=True,
    )

    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frequency_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    presence_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    system_prompt_template: Mapped[str | None] = mapped_column(Text, nullable=True)

    json_mode_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    tool_choice_default: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stream_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    extra_parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    ai_model: Mapped[AIModel] = relationship(
        "AIModel",
        back_populates="configuration",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<AIModelConfiguration id={self.id} ai_model_id={self.ai_model_id}>"
        )
