"""
Shared SQLAlchemy declarative base for all database models.

Per the "Extend, Don't Rewrite" engineering principle, every ORM model in
the codebase must inherit from a single declarative base so all table
metadata lands on one shared ``sqlalchemy.orm.DeclarativeBase.metadata``
object — introducing a second, parallel declarative base would give some
models their own isolated metadata, making them invisible to Alembic's
autogenerate diffing and to relationships defined against the "main" base.

The actual ``Base`` class is defined once in ``src/models/base.py``
(alongside the ORM layer it anchors); this module re-exports that same
object under ``core`` so infrastructure-facing code that lives in or reads
this package can reference ``src.core.base.Base`` as a stable, documented
import path without needing to know it is physically defined in ``models``.
It is a plain re-export — no new class or behavior is introduced here, so
``src.core.base.Base is src.models.base.Base`` always holds.
"""

from src.models.base import Base

__all__ = ["Base"]

