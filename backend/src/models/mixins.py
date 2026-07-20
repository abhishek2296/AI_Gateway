"""Reusable column groups shared across multiple ORM models."""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """
    Adds created_at and updated_at audit columns to any model that inherits it.

    Server defaults keep timestamps consistent when rows are inserted outside
    the application (migrations, admin scripts). The ORM-level onupdate keeps
    updated_at current on every UPDATE issued through SQLAlchemy.
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
