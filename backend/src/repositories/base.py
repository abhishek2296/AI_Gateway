"""Generic async repository base for SQLAlchemy 2.x persistence."""

from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from src.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Reusable CRUD operations for a single ORM entity type.

    Repositories flush when IDs or server defaults are needed but never
    commit or rollback — transaction boundaries belong to the Unit of Work
    layer (Phase 3.9).
    """

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        self._session = session
        self._model = model

    @property
    def session(self) -> AsyncSession:
        """Expose the bound session for advanced queries in subclasses."""
        return self._session

    async def create(self, entity: ModelT) -> ModelT:
        """Persist a new entity and return it with generated keys loaded."""
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: int) -> ModelT | None:
        """Fetch one row by primary key."""
        return await self._session.get(self._model, entity_id)

    async def get_one(
        self,
        *filters: ColumnElement[bool],
        options: Sequence[Any] | None = None,
    ) -> ModelT | None:
        """Return the first row matching filters, or ``None``."""
        stmt = self._select(*filters, options=options).limit(1)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        *filters: ColumnElement[bool],
        order_by: Sequence[ColumnElement[Any] | InstrumentedAttribute[Any]] | None = None,
        offset: int | None = None,
        limit: int | None = None,
        options: Sequence[Any] | None = None,
    ) -> list[ModelT]:
        """Return rows matching filters with optional ordering and pagination."""
        stmt = self._select(*filters, order_by=order_by, offset=offset, limit=limit, options=options)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, entity: ModelT, **values: Any) -> ModelT:
        """Apply in-place attribute updates and flush."""
        for key, value in values.items():
            setattr(entity, key, value)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def delete(self, entity: ModelT) -> None:
        """Remove an entity from the current session."""
        await self._session.delete(entity)
        await self._session.flush()

    async def delete_by_id(self, entity_id: int) -> bool:
        """Delete by primary key. Returns ``False`` when the row is missing."""
        entity = await self.get_by_id(entity_id)
        if entity is None:
            return False
        await self.delete(entity)
        return True

    async def exists(self, *filters: ColumnElement[bool]) -> bool:
        """Return whether any row matches the given filters."""
        return await self.count(*filters) > 0

    async def count(self, *filters: ColumnElement[bool]) -> int:
        """Count rows, optionally filtered."""
        stmt = select(func.count()).select_from(self._model)
        if filters:
            stmt = stmt.where(*filters)
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    def _select(
        self,
        *filters: ColumnElement[bool],
        order_by: Sequence[ColumnElement[Any] | InstrumentedAttribute[Any]] | None = None,
        offset: int | None = None,
        limit: int | None = None,
        options: Sequence[Any] | None = None,
    ) -> Select[tuple[ModelT]]:
        """Build a typed ``select`` for this repository's model."""
        stmt = select(self._model)
        if filters:
            stmt = stmt.where(*filters)
        if options:
            stmt = stmt.options(*options)
        if order_by:
            stmt = stmt.order_by(*order_by)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt
