"""
ORM entity for per-request usage analytics and billing.

Every gateway request that reaches a provider produces (or attempts to
produce) exactly one ``UsageRecord`` row, regardless of whether it
succeeded. This is the system's audit trail for token consumption, latency,
and estimated cost — it must never be silently lost or overwritten, which
is why related foreign keys favor ``SET NULL``/``RESTRICT`` over ``CASCADE``
(see column docs below) and why this entity has no update-oriented service
methods; once a record is written, it is treated as immutable history.
"""

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

    Maps to the ``usage_records`` table. Tracks token counts, latency, and
    estimated cost for analytics and billing. ``request_id`` correlates with
    the HTTP ``X-Request-ID`` middleware value and is unique to prevent
    duplicate billing rows if a request is retried (e.g. client retry after
    a timeout that actually succeeded server-side) — the service layer
    should upsert/ignore-on-conflict against this uniqueness rather than
    blindly inserting on every attempt. ``chat_session_id`` is optional for
    stateless or non-session requests (e.g. a one-off completion call not
    tied to any conversation).

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        chat_session_id: Optional foreign key to ``chat_sessions.id``.
            ``ondelete="SET NULL"`` — deleting a session must NOT delete
            its usage history; the FK is nulled out instead so billing
            records survive session deletion. ``nullable=True`` because not
            every request originates from a persisted chat session.
            Indexed.
        provider_id: Foreign key to ``providers.id``. ``ondelete="RESTRICT"``
            — a provider with existing usage records cannot be deleted at
            all, since silently losing billing history (via ``CASCADE``) or
            orphaning it (via ``SET NULL``) are both unacceptable for
            financial/audit data. Indexed.
        ai_model_id: Foreign key to ``ai_models.id``. Same ``RESTRICT``
            rationale as ``provider_id`` — usage history must not be
            deletable or orphaned by removing the model it references.
            Indexed.
        request_id: Correlates this record with the gateway's
            ``X-Request-ID`` for a given HTTP call. ``String(64)``,
            indexed, and constrained unique via ``__table_args__`` to
            guard against duplicate billing rows from retried requests.
        prompt_tokens: Number of input/prompt tokens consumed, or ``None``
            if the provider didn't report token usage. ``Integer``,
            nullable.
        completion_tokens: Number of output/completion tokens generated,
            or ``None`` if not reported. ``Integer``, nullable.
        total_tokens: Total tokens (prompt + completion) for this request,
            or ``None`` if not reported. ``Integer``, nullable. Stored
            explicitly rather than always computed, since some providers
            report a total that doesn't exactly equal the sum of the two
            (e.g. due to internal overhead tokens).
        estimated_cost: Estimated monetary cost of this request, computed
            from token counts and the provider's pricing. ``Numeric(12, 6)``
            for sub-cent precision at scale, nullable if cost couldn't be
            computed (e.g. unknown pricing for a self-hosted model).
        latency_ms: Total round-trip latency for this request in
            milliseconds, or ``None`` if not measured. ``Integer``,
            nullable.
        status: Outcome of the request, e.g. ``"success"``, ``"error"``,
            ``"timeout"``. ``String(50)``, indexed, following the
            string-identity pattern (ADR-007) rather than a database enum.
        error_message: Optional human-readable error detail when
            ``status`` indicates failure. ``Text``, nullable — should never
            contain secrets/stack traces per ``08-security.mdc``.
        request_timestamp: When the underlying request was made (as
            opposed to ``created_at``, which is when this audit row itself
            was inserted — these are usually the same instant but are
            conceptually distinct, e.g. for backfilled records).
            ``DateTime(timezone=True)``, indexed for time-range queries.
        chat_session: Many-to-one (optional) back to
            :class:`~src.models.chat_session.ChatSession`. Eagerly loaded
            (``lazy="selectin"``); may be ``None``.
        provider: Many-to-one back to :class:`~src.models.provider.Provider`.
            Eagerly loaded (``lazy="selectin"``).
        ai_model: Many-to-one back to :class:`~src.models.ai_model.AIModel`.
            Eagerly loaded (``lazy="selectin"``).

    Example:
        >>> from datetime import datetime, timezone
        >>> record = UsageRecord(
        ...     provider_id=1,
        ...     ai_model_id=1,
        ...     request_id="req-abc123",
        ...     prompt_tokens=120,
        ...     completion_tokens=45,
        ...     total_tokens=165,
        ...     status="success",
        ...     request_timestamp=datetime.now(timezone.utc),
        ... )
        >>> record.status
        'success'
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
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<UsageRecord id=1 request_id='req-abc123'
            status='success'>``. Token counts and cost are omitted for
            brevity; ``request_id`` and ``status`` are the most useful
            fields for quickly spotting a record in logs.

        Example:
            >>> UsageRecord(id=1, request_id="req-abc123", status="success").__repr__()
            "<UsageRecord id=1 request_id='req-abc123' status='success'>"
        """
        return (
            f"<UsageRecord id={self.id} request_id={self.request_id!r} "
            f"status={self.status!r}>"
        )
