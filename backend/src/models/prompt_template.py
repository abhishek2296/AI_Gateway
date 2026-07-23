"""ORM entity for reusable prompt templates."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Integer, String, Text, UniqueConstraint, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base
from src.models.mixins import TimestampMixin


class PromptTemplate(Base, TimestampMixin):
    """
    Versioned, reusable prompt template for gateway workloads.

    ``variables`` holds a JSON schema or default values for template
    placeholders. No FK to :class:`~src.models.chat_session.ChatSession`
    in this phase — linking templates to sessions belongs in the API layer.
    """

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "version",
            name="uq_prompt_templates_name_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    template: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")

    variables: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<PromptTemplate id={self.id} name={self.name!r} "
            f"version={self.version}>"
        )
