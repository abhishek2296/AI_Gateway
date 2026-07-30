"""
Reusable column groups shared across multiple ORM models.

Rather than redeclaring the same audit columns on every entity in
``src/models/``, common column groups are factored into mixin classes here.
A model opts in by adding the mixin to its base classes, e.g.
``class Provider(Base, TimestampMixin): ...``. Because Python resolves
attributes via MRO, the mixin's ``Mapped``/``mapped_column`` declarations are
picked up by SQLAlchemy's declarative machinery exactly as if they had been
written directly on the model class.
"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """
    Adds ``created_at`` and ``updated_at`` audit columns to any model that inherits it.

    Server defaults keep timestamps consistent when rows are inserted outside
    the application (migrations, admin scripts, raw SQL) — the database
    itself stamps the time rather than relying on application code to set it.
    The ORM-level ``onupdate`` additionally keeps ``updated_at`` current on
    every ``UPDATE`` issued *through SQLAlchemy*; it does not fire for
    updates made outside the ORM (e.g. raw SQL, bulk updates), which is a
    known trade-off of this approach.

    This is a plain mixin (not itself a mapped class and not a subclass of
    :class:`~src.models.base.Base`) — it only supplies column declarations
    that a concrete model combines with ``Base``.

    Attributes:
        created_at: Timestamp (timezone-aware) recorded when the row is
            first inserted. Never ``NULL``. Defaults to the database's
            current time (``func.now()``) at insert time if the application
            does not supply a value explicitly.
        updated_at: Timestamp (timezone-aware) recorded when the row is
            inserted, and refreshed automatically on every subsequent
            ORM-issued ``UPDATE``. Never ``NULL``.

    Example:
        >>> from src.models.base import Base
        >>> class Widget(Base, TimestampMixin):
        ...     __tablename__ = "widgets"
        ...     id: Mapped[int] = mapped_column(primary_key=True)
        >>> "created_at" in Widget.__table__.columns
        True
        >>> "updated_at" in Widget.__table__.columns
        True
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
