"""
Unit of Work — transaction boundary for the persistence layer.

Re-exports the public surface of the ``unit_of_work`` package so callers
elsewhere in the codebase (e.g. ``services/``) can write
``from src.unit_of_work import AsyncUnitOfWork`` instead of reaching into
the submodules directly:

- ``BaseUnitOfWork``  — abstract contract (``unit_of_work.base``).
- ``AsyncUnitOfWork``  — concrete SQLAlchemy-backed implementation
  (``unit_of_work.unit_of_work``).
"""

from src.unit_of_work.base import BaseUnitOfWork
from src.unit_of_work.unit_of_work import AsyncUnitOfWork

__all__ = [
    "BaseUnitOfWork",
    "AsyncUnitOfWork",
]
