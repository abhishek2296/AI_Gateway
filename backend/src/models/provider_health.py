"""ORM entity for provider health check snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.provider import Provider


class ProviderHealth(Base, TimestampMixin):
    """
    Point-in-time health snapshot for a :class:`~src.models.provider.Provider`.

    Each row records one probe result. Historical rows support trend analysis;
    the latest row (by ``checked_at``) represents current provider health.
    ``status`` values follow string identity (e.g. ``healthy``, ``unhealthy``,
    ``degraded``) — consistent with ADR-007.
    """

    __tablename__ = "provider_health"

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    status: Mapped[str] = mapped_column(String(50), index=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_failure_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    health_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="health_checks",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<ProviderHealth id={self.id} provider_id={self.provider_id} "
            f"status={self.status!r} checked_at={self.checked_at!s}>"
        )
