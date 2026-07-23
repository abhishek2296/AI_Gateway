"""Abstract Unit of Work interface for persistence transaction boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self


class BaseUnitOfWork(ABC):
    """
    Coordinates repositories and owns the database transaction lifecycle.

    Concrete implementations expose entity repositories and implement
    ``commit``, ``rollback``, and ``close``. Use as an async context manager
    to roll back on exception and release the session — commit remains explicit.
    """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self.rollback()
        await self.close()

    @abstractmethod
    async def commit(self) -> None:
        """Persist all changes in the current transaction."""

    @abstractmethod
    async def rollback(self) -> None:
        """Discard uncommitted changes in the current transaction."""

    @abstractmethod
    async def close(self) -> None:
        """Release the underlying database session."""
