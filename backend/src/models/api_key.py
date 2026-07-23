"""ORM entity for provider API credentials (env-var references only)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint, false, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.provider import Provider


class APIKey(Base, TimestampMixin):
    """
    Credential reference for a :class:`~src.models.provider.Provider`.

    ``api_key_env`` stores the **environment variable name** (e.g.
    ``OPENAI_API_KEY``), never the secret value. ``key_identifier`` is a
    non-secret label for admin UI (e.g. ``prod-primary``, ``last4:…8f2a``).
    At most one row per provider may set ``is_default=True``.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("provider_id", "name", name="uq_api_keys_provider_id_name"),
        Index(
            "uq_api_keys_one_default_per_provider",
            "provider_id",
            unique=True,
            postgresql_where=text("is_default IS TRUE"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255))
    key_identifier: Mapped[str] = mapped_column(String(255), index=True)

    api_key_env: Mapped[str] = mapped_column(String(255))
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True)
    project: Mapped[str | None] = mapped_column(String(255), nullable=True)

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

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="api_keys",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<APIKey id={self.id} name={self.name!r} provider_id={self.provider_id}>"
        )
