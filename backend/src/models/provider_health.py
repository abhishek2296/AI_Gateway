"""
ORM entity for provider health check snapshots.

Each row is one point-in-time probe result for a
:class:`~src.models.provider.Provider` (e.g. from a periodic background
health-check job). Storing every check as its own row — rather than
overwriting a single "current status" column on ``Provider`` — preserves a
full history that supports uptime/trend analysis and post-incident review,
at the cost of the table growing unboundedly (retention/pruning is left to
a future operational concern, not modeled here).
"""

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

    Maps to the ``provider_health`` table. Each row records one probe
    result. Historical rows support trend analysis; the latest row (by
    ``checked_at``, not by ``id`` — health checks could in principle be
    backfilled or arrive out of insertion order) represents current
    provider health. ``status`` values follow string identity (e.g.
    ``"healthy"``, ``"unhealthy"``, ``"degraded"``) — consistent with
    ADR-007 — rather than a database enum, so new statuses don't require a
    migration.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        provider_id: Foreign key to ``providers.id``. ``ondelete="CASCADE"``
            — health history has no meaning without its provider, so it is
            purged automatically when the provider is deleted (unlike
            usage records, health checks are operational telemetry, not
            financial/audit data, so cascading deletion is acceptable
            here). Indexed.
        status: Outcome of this probe, e.g. ``"healthy"``, ``"unhealthy"``,
            ``"degraded"``. ``String(50)``, indexed.
        latency_ms: Response time observed during this probe, in
            milliseconds, or ``None`` if the probe failed before a
            response was received. ``Integer``, nullable.
        last_success_at: Timestamp of the most recent *successful* probe
            as known at the time this row was written, or ``None`` if
            there has never been a success. ``DateTime(timezone=True)``,
            nullable. This is a rolling summary value copied onto each
            row (not derived purely from this row's own ``checked_at``),
            which lets consumers read "time since last success" from a
            single row without scanning history.
        last_failure_at: Timestamp of the most recent *failed* probe as
            known at the time this row was written, or ``None`` if there
            has never been a failure. Same rolling-summary rationale as
            ``last_success_at``. ``DateTime(timezone=True)``, nullable.
        failure_reason: Human-readable explanation when this probe failed
            (e.g. ``"connection timeout"``), or ``None`` on success.
            ``Text``, nullable.
        health_score: Optional normalized health metric (e.g. a rolling
            success-rate percentage or composite score), or ``None`` if
            not computed. ``Float``, nullable — the scoring algorithm
            itself lives in the service layer, not this model.
        checked_at: When this probe was actually performed (as opposed to
            ``created_at``, which is when the row was inserted — normally
            the same instant, but kept distinct for backfilled or
            batched writes). ``DateTime(timezone=True)``, indexed, and
            used as the ordering key for
            :attr:`~src.models.provider.Provider.health_checks`.
        provider: Many-to-one back to :class:`~src.models.provider.Provider`.
            Eagerly loaded (``lazy="selectin"``).

    Example:
        >>> from datetime import datetime, timezone
        >>> check = ProviderHealth(
        ...     provider_id=1,
        ...     status="healthy",
        ...     latency_ms=42,
        ...     checked_at=datetime.now(timezone.utc),
        ... )
        >>> check.status
        'healthy'
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
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<ProviderHealth id=1 provider_id=1
            status='healthy' checked_at=2026-07-30 12:00:00+00:00>``,
            identifying the provider, outcome, and probe time at a glance.

        Example:
            >>> from datetime import datetime, timezone
            >>> ph = ProviderHealth(
            ...     id=1,
            ...     provider_id=1,
            ...     status="healthy",
            ...     checked_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ... )
            >>> ph.__repr__()
            "<ProviderHealth id=1 provider_id=1 status='healthy' checked_at=2026-01-01 00:00:00+00:00>"
        """
        return (
            f"<ProviderHealth id={self.id} provider_id={self.provider_id} "
            f"status={self.status!r} checked_at={self.checked_at!s}>"
        )
