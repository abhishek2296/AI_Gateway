"""
Single declarative registry for every SQLAlchemy ORM model in this service.

This module exists purely to define the shared :class:`Base` class. Every
ORM entity in ``src/models/`` (``Provider``, ``AIModel``, ``ChatSession``,
etc.) inherits from it so all table metadata lands on one
``sqlalchemy.orm.DeclarativeBase.metadata`` object. That single, shared
``MetaData`` instance is what Alembic's autogeneration (``alembic/env.py``)
and the async engine (``core/database.py``) introspect when creating tables,
comparing schema state, or generating migrations — if models used different
``Base`` classes, each would get its own isolated metadata and Alembic would
only "see" whichever one it was pointed at.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Root mapper base for all database tables.

    Every ORM model must inherit from this class so SQLAlchemy registers
    its table on one shared ``MetaData`` instance. Alembic migrations and the
    async engine both rely on that single registry when reflecting or
    creating schema.

    This class intentionally has no columns or behavior of its own — it is a
    pure marker/registry base. Shared, reusable *columns* (like audit
    timestamps) belong in mixins such as
    :class:`~src.models.mixins.TimestampMixin`, which are combined with this
    base via multiple inheritance on each concrete model
    (e.g. ``class Provider(Base, TimestampMixin): ...``).

    Example:
        >>> from sqlalchemy.orm import Mapped, mapped_column
        >>> class Widget(Base):
        ...     __tablename__ = "widgets"
        ...     id: Mapped[int] = mapped_column(primary_key=True)
        >>> "widgets" in Base.metadata.tables
        True
    """
