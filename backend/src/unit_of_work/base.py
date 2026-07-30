"""
Abstract Unit of Work interface for persistence transaction boundaries.

The Unit of Work (UoW) pattern gives the service layer a single object that
owns a database transaction and exposes the repositories that participate in
it, so multiple repository operations can be grouped into one atomic commit
(or rolled back together on failure) instead of each repository managing its
own transaction independently. ``BaseUnitOfWork`` defines the contract every
concrete UoW (e.g. ``unit_of_work.AsyncUnitOfWork``) must implement; services
depend on this abstract type rather than a specific backend so the
persistence engine can change without touching business logic (per the
Layer Boundaries rule: `services/` must not be coupled to a specific ORM
session implementation).
"""

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

    Commit is intentionally never automatic: only the caller that finished a
    coherent unit of work (e.g. "create a chat session and its first
    message") knows the right moment to persist, so ``__aexit__`` only
    handles the failure path (rollback + close) and always leaves ``commit``
    up to the caller.

    Example:
        >>> class _NoOpUnitOfWork(BaseUnitOfWork):
        ...     async def commit(self) -> None: pass
        ...     async def rollback(self) -> None: pass
        ...     async def close(self) -> None: pass
        >>> import asyncio
        >>> async def _demo():
        ...     async with _NoOpUnitOfWork() as uow:
        ...         await uow.commit()
        >>> asyncio.run(_demo())
    """

    async def __aenter__(self) -> Self:
        """
        Enter the async context manager, returning this UoW for use in a ``with`` block.

        Returns:
            This same ``BaseUnitOfWork`` instance (``Self``), so callers can
            write ``async with SomeUnitOfWork(session) as uow: ...`` and then
            access repositories via ``uow``.

        Example:
            >>> import asyncio
            >>> class _NoOpUnitOfWork(BaseUnitOfWork):
            ...     async def commit(self) -> None: pass
            ...     async def rollback(self) -> None: pass
            ...     async def close(self) -> None: pass
            >>> async def _demo():
            ...     uow = _NoOpUnitOfWork()
            ...     entered = await uow.__aenter__()
            ...     return entered is uow
            >>> asyncio.run(_demo())
            True
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Roll back on exception, then always release the session.

        This is the safety net for the transaction: if the code inside the
        ``async with`` block raised (``exc_type`` is not ``None``), any
        partially-applied changes must not be persisted, so ``rollback`` runs
        before the session is closed. If the block completed without an
        exception, whether anything was actually persisted depends entirely
        on whether the caller already awaited ``commit`` — this method never
        commits, only cleans up.

        Args:
            exc_type: The exception class raised inside the ``async with``
                block, or ``None`` if it completed normally.
            exc_val: The exception instance raised, or ``None``.
            exc_tb: The exception's traceback, or ``None``.

        Returns:
            None. Returning ``None`` (rather than a truthy value) means any
            exception that occurred is *not* suppressed — it continues to
            propagate to the caller after cleanup runs, which is the correct
            behavior for a resource-management context manager.

        Raises:
            Exception: Whatever ``self.rollback()`` or ``self.close()``
                raise, if the underlying session/connection fails during
                cleanup; the original exception (if any) is not swallowed.

        Example:
            >>> import asyncio
            >>> class _TrackingUnitOfWork(BaseUnitOfWork):
            ...     def __init__(self):
            ...         self.rolled_back = False
            ...     async def commit(self) -> None: pass
            ...     async def rollback(self) -> None:
            ...         self.rolled_back = True
            ...     async def close(self) -> None: pass
            >>> async def _demo():
            ...     uow = _TrackingUnitOfWork()
            ...     try:
            ...         async with uow:
            ...             raise ValueError("boom")
            ...     except ValueError:
            ...         pass
            ...     return uow.rolled_back
            >>> asyncio.run(_demo())
            True
        """
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
