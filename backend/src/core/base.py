"""Shared SQLAlchemy declarative base for all database models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class that turns Python model classes into database table mappings."""

