"""Single declarative registry for every SQLAlchemy ORM model in this service."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Root mapper base for all database tables.

    Every ORM model must inherit from this class so SQLAlchemy registers
    its table on one shared MetaData instance. Alembic migrations and the
    async engine both rely on that single registry when reflecting or
    creating schema.
    """
