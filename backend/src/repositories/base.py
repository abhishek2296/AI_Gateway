"""
Generic async repository base for SQLAlchemy 2.x persistence.

This module defines :class:`BaseRepository`, the single shared implementation
of "boring" CRUD operations (create, get, list, update, delete, count,
exists) that every entity-specific repository in ``src/repositories/``
inherits from. The goal is that a concrete repository (e.g.
``ProviderRepository``) only needs to implement the queries that are
*specific* to its entity — anything generic is written once here.

This is the data-access layer boundary described in
``01-engineering-principles.mdc``: repositories talk to the database via
SQLAlchemy's async ORM and return ORM entities, but they never commit or
roll back a transaction themselves. Transaction boundaries belong to the
Unit of Work layer (Phase 3.9); repositories only ``flush()`` when they need
generated primary keys or server-side defaults (e.g. ``created_at``) to be
populated on the Python object before returning it.
"""

from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from src.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)
"""Type variable bound to :class:`~src.models.base.Base`.

Every concrete repository parameterizes :class:`BaseRepository` with the
specific ORM model it manages, e.g. ``BaseRepository[Provider]``. This lets
type checkers (and IDEs) know that ``ProviderRepository.get_by_id`` returns
``Provider | None`` rather than a generic ``Base | None``.
"""


class BaseRepository(Generic[ModelT]):
    """
    Reusable CRUD operations for a single ORM entity type.

    ``BaseRepository`` is the generic parent class for every repository in
    this package (see ``src/repositories/__init__.py`` for the full list).
    Subclasses call ``super().__init__(session, SomeModel)`` to bind
    themselves to one SQLAlchemy model and then add entity-specific query
    methods (e.g. ``ProviderRepository.get_by_name``) on top of the generic
    operations defined here (``create``, ``get_by_id``, ``get_one``,
    ``list``, ``update``, ``delete``, ``delete_by_id``, ``exists``,
    ``count``).

    Repositories flush when IDs or server defaults are needed but never
    commit or rollback — transaction boundaries belong to the Unit of Work
    layer (Phase 3.9). This means callers (services, Unit of Work) are
    responsible for calling ``session.commit()`` once all repository calls
    for a logical operation have succeeded.

    Example:
        >>> class ProviderRepository(BaseRepository[Provider]):
        ...     def __init__(self, session: AsyncSession) -> None:
        ...         super().__init__(session, Provider)
        ...
        ...     async def get_by_name(self, name: str) -> Provider | None:
        ...         return await self.get_one(Provider.name == name)
    """

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        """
        Bind this repository to a live session and the ORM model it manages.

        Args:
            session: The active ``AsyncSession`` this repository issues
                queries through. The same session must be shared by every
                repository participating in one logical unit of work so
                that they see each other's uncommitted (flushed) changes.
            model: The SQLAlchemy declarative model class this repository
                is responsible for (e.g. ``Provider``). Stored so generic
                methods like ``get_by_id``/``list``/``count`` know which
                table to query without subclasses repeating it.

        Example:
            >>> repo = BaseRepository(session, Provider)
        """
        self._session = session
        self._model = model

    @property
    def session(self) -> AsyncSession:
        """
        Expose the bound session for advanced queries in subclasses.

        Subclasses use this when they need to build a query that the
        generic ``_select``/``list``/``get_one`` helpers cannot express,
        e.g. a multi-table ``join`` (see
        ``ProviderRepository.get_default_provider``).

        Returns:
            The ``AsyncSession`` this repository was constructed with.

        Example:
            >>> await repo.session.execute(some_custom_statement)
        """
        return self._session

    async def create(self, entity: ModelT) -> ModelT:
        """
        Persist a new entity and return it with generated keys loaded.

        The entity is added to the session and flushed (not committed), then
        refreshed so that database-generated values — the primary key,
        ``server_default`` columns, and any ``TimestampMixin`` timestamps —
        are populated on the Python object the caller already holds a
        reference to.

        Args:
            entity: A transient (not-yet-persisted) instance of ``ModelT``,
                constructed with whatever fields the caller has already set
                (e.g. ``Provider(name="openai", display_name="OpenAI")``).

        Returns:
            The same ``entity`` instance, now attached to the session with
            its primary key and server-generated defaults populated.

        Raises:
            sqlalchemy.exc.IntegrityError: If the flush violates a database
                constraint (e.g. a unique or foreign-key constraint on the
                target table).

        Example:
            >>> provider = Provider(name="openai", display_name="OpenAI", provider_type="openai")
            >>> saved = await repo.create(provider)
            >>> saved.id is not None
            True
        """
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: int) -> ModelT | None:
        """
        Fetch one row by primary key.

        Uses ``AsyncSession.get``, which first checks the session's identity
        map (in-memory cache of already-loaded objects) before issuing a
        query, so repeated lookups of the same id within one unit of work
        are cheap.

        Args:
            entity_id: The primary key value to look up.

        Returns:
            The matching ``ModelT`` instance, or ``None`` if no row with
            that primary key exists.

        Example:
            >>> provider = await repo.get_by_id(1)
            >>> provider is None or provider.id == 1
            True
        """
        return await self._session.get(self._model, entity_id)

    async def get_one(
        self,
        *filters: ColumnElement[bool],
        options: Sequence[Any] | None = None,
    ) -> ModelT | None:
        """
        Return the first row matching filters, or ``None``.

        This is the building block subclasses use for "fetch by unique
        key" style lookups (e.g. ``ProviderRepository.get_by_name``). A
        ``LIMIT 1`` is applied so it is safe to call even against a filter
        that is not guaranteed unique at the database level; only the first
        matching row (in whatever order the database happens to return
        rows) is returned.

        Args:
            *filters: Zero or more SQLAlchemy boolean column expressions
                (e.g. ``Provider.name == "openai"``), combined with ``AND``
                via ``Select.where``.
            options: Optional loader options (e.g.
                ``[selectinload(ChatSession.messages)]``) to eagerly load
                relationships on the returned row, avoiding N+1 queries.

        Returns:
            The first matching ``ModelT``, or ``None`` if no row matches.

        Example:
            >>> provider = await repo.get_one(Provider.name == "openai")
            >>> provider is None or provider.name == "openai"
            True
        """
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
        """
        Return rows matching filters with optional ordering and pagination.

        This is the shared implementation behind most subclass "list"
        methods (e.g. ``MessageRepository.list_messages``,
        ``ProviderRepository.list_active``). Subclasses typically hardcode
        one or more ``filters``/``order_by`` values that are specific to
        the entity and simply forward ``offset``/``limit`` from their own
        signature.

        Args:
            *filters: Zero or more SQLAlchemy boolean column expressions,
                combined with ``AND``.
            order_by: Columns (or ``InstrumentedAttribute`` references) to
                sort by, applied in the given order, e.g.
                ``(Provider.name.asc(),)``. ``None`` leaves ordering
                undefined (database default, typically insertion order).
            offset: Number of matching rows to skip, for pagination.
                ``None`` means no offset.
            limit: Maximum number of rows to return. ``None`` means no
                limit (return every matching row).
            options: Optional loader options for eager-loading
                relationships, forwarded to ``Select.options``.

        Returns:
            A list of matching ``ModelT`` instances, in the requested order.
            Empty list (never ``None``) when nothing matches.

        Example:
            >>> providers = await repo.list(
            ...     Provider.is_active.is_(True),
            ...     order_by=(Provider.name.asc(),),
            ...     limit=10,
            ... )
            >>> isinstance(providers, list)
            True
        """
        stmt = self._select(*filters, order_by=order_by, offset=offset, limit=limit, options=options)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, entity: ModelT, **values: Any) -> ModelT:
        """
        Apply in-place attribute updates and flush.

        Values are set via ``setattr`` rather than a bulk ``UPDATE``
        statement so that SQLAlchemy's ORM-level change tracking,
        validators, and ``onupdate``/``TimestampMixin`` hooks (e.g.
        auto-updating ``updated_at``) all run normally, exactly as if the
        caller had set the attributes directly.

        Args:
            entity: An already-persisted (attached to the session) instance
                to modify.
            **values: Column-name/value pairs to assign onto ``entity``,
                e.g. ``update(session, is_archived=True)``.

        Returns:
            The same ``entity`` instance, refreshed so any
            database-computed values (e.g. ``updated_at``) reflect the
            just-applied change.

        Raises:
            sqlalchemy.exc.IntegrityError: If the flush violates a database
                constraint (e.g. a unique constraint on one of the updated
                columns).

        Example:
            >>> archived = await repo.update(session_entity, is_archived=True)
            >>> archived.is_archived
            True
        """
        for key, value in values.items():
            setattr(entity, key, value)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def delete(self, entity: ModelT) -> None:
        """
        Remove an entity from the current session.

        Args:
            entity: An already-persisted instance to delete. Any
                relationships configured with ``cascade="all, delete-orphan"``
                on the model (see e.g. ``Provider.ai_models``) will have
                their dependent rows deleted as part of the same flush.

        Returns:
            None. The deletion is flushed to the database but not
            committed; the row disappears from subsequent queries within
            this session but is only durable once the surrounding
            transaction commits.

        Example:
            >>> await repo.delete(entity)
        """
        await self._session.delete(entity)
        await self._session.flush()

    async def delete_by_id(self, entity_id: int) -> bool:
        """
        Delete by primary key. Returns ``False`` when the row is missing.

        This is a convenience wrapper around ``get_by_id`` + ``delete`` so
        callers that only have an id (e.g. from a route path parameter)
        don't need to fetch the entity themselves first.

        Args:
            entity_id: The primary key of the row to delete.

        Returns:
            ``True`` if a matching row was found and deleted, ``False`` if
            no row with that id existed (a no-op — nothing is raised).

        Example:
            >>> await repo.delete_by_id(999)  # no such row
            False
        """
        entity = await self.get_by_id(entity_id)
        if entity is None:
            return False
        await self.delete(entity)
        return True

    async def exists(self, *filters: ColumnElement[bool]) -> bool:
        """
        Return whether any row matches the given filters.

        Implemented in terms of ``count`` rather than a separate
        ``EXISTS`` query for simplicity; the count is cheap for the small,
        indexed lookups this method is used for (e.g. uniqueness checks
        before insert).

        Args:
            *filters: Zero or more SQLAlchemy boolean column expressions,
                combined with ``AND``.

        Returns:
            ``True`` if at least one row matches, ``False`` otherwise.

        Example:
            >>> await repo.exists(Provider.name == "openai")
            True
        """
        return await self.count(*filters) > 0

    async def count(self, *filters: ColumnElement[bool]) -> int:
        """
        Count rows, optionally filtered.

        Uses ``SELECT count(*)`` rather than loading and counting entities
        in Python, so this stays cheap even for large tables.

        Args:
            *filters: Zero or more SQLAlchemy boolean column expressions to
                narrow the count, combined with ``AND``. With no filters,
                counts every row in the table.

        Returns:
            The number of matching rows as a plain ``int``.

        Example:
            >>> await repo.count(Provider.is_active.is_(True))
            3
        """
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
        """
        Build a typed ``select`` for this repository's model.

        Internal helper shared by ``get_one`` and ``list`` so the clause
        ordering (``WHERE`` → loader ``options`` → ``ORDER BY`` →
        ``OFFSET`` → ``LIMIT``) is only written once and stays consistent
        between both public methods.

        Args:
            *filters: Zero or more SQLAlchemy boolean column expressions,
                combined with ``AND`` via ``Select.where``.
            order_by: Columns to sort by, applied in the given order.
                ``None`` or empty leaves ordering unset.
            offset: Number of rows to skip. ``None`` applies no offset.
            limit: Maximum number of rows to return. ``None`` applies no
                limit.
            options: Loader options (e.g. ``selectinload(...)``) applied via
                ``Select.options`` to control eager loading of relationships.

        Returns:
            A fully constructed, not-yet-executed ``Select`` statement
            scoped to ``self._model``.

        Example:
            >>> stmt = repo._select(Provider.is_active.is_(True), limit=5)
            >>> result = await repo.session.execute(stmt)
        """
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
