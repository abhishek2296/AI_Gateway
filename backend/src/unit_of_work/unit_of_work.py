"""
Async SQLAlchemy Unit of Work coordinating all entity repositories.

Concrete implementation of ``unit_of_work.base.BaseUnitOfWork`` backed by a
single SQLAlchemy ``AsyncSession``. Every repository exposed here is
constructed with that same session instance, which is what makes the
"unit of work" atomic: writes made through ``uow.providers``,
``uow.chat_sessions``, etc. all participate in one transaction and are
flushed/committed together via a single ``await uow.commit()``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.ai_model_configuration_repository import AIModelConfigurationRepository
from src.repositories.ai_model_repository import AIModelRepository
from src.repositories.api_key_repository import APIKeyRepository
from src.repositories.chat_session_repository import ChatSessionRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.prompt_template_repository import PromptTemplateRepository
from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
from src.repositories.provider_health_repository import ProviderHealthRepository
from src.repositories.provider_repository import ProviderRepository
from src.repositories.usage_record_repository import UsageRecordRepository
from src.unit_of_work.base import BaseUnitOfWork


class AsyncUnitOfWork(BaseUnitOfWork):
    """
    Unit of Work backed by a single :class:`AsyncSession`.

    All repositories share the same session so multiple entities can be
    persisted atomically via ``await uow.commit()``.

    Example::

        async with AsyncUnitOfWork(session) as uow:
            provider = await uow.providers.get_by_name("ollama")
            await uow.commit()

    On exception the context manager rolls back and closes the session.
    Commit is never automatic.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        close_session: bool = True,
    ) -> None:
        """
        Bind this Unit of Work to a session and construct all entity repositories.

        Every repository is instantiated up front (rather than lazily) with
        the *same* ``session`` object, which is the mechanism that ties all
        of their reads/writes into one shared transaction.

        Args:
            session: The ``AsyncSession`` this Unit of Work — and every
                repository it exposes — will use for all queries and writes.
            close_session: Whether ``close()`` should actually close the
                underlying session. Defaults to ``True``. Set to ``False``
                when the session's lifecycle is managed externally (e.g. by
                a request-scoped dependency that closes it itself), so this
                Unit of Work doesn't close a session another caller still
                needs.

        Example:
            >>> from unittest.mock import MagicMock
            >>> from sqlalchemy.ext.asyncio import AsyncSession
            >>> session = MagicMock(spec=AsyncSession)
            >>> uow = AsyncUnitOfWork(session)
            >>> uow.session is session
            True
        """
        self._session = session
        self._close_session = close_session
        self._closed = False

        self._providers = ProviderRepository(session)
        self._ai_models = AIModelRepository(session)
        self._provider_configurations = ProviderConfigurationRepository(session)
        self._ai_model_configurations = AIModelConfigurationRepository(session)
        self._chat_sessions = ChatSessionRepository(session)
        self._messages = MessageRepository(session)
        self._prompt_templates = PromptTemplateRepository(session)
        self._api_keys = APIKeyRepository(session)
        self._usage_records = UsageRecordRepository(session)
        self._provider_health = ProviderHealthRepository(session)

    @property
    def session(self) -> AsyncSession:
        """Underlying session for advanced queries outside repositories."""
        return self._session

    @property
    def providers(self) -> ProviderRepository:
        """Repository for provider catalog rows (``Provider``), sharing this UoW's session."""
        return self._providers

    @property
    def ai_models(self) -> AIModelRepository:
        """Repository for registered LLM models (``AIModel``), sharing this UoW's session."""
        return self._ai_models

    @property
    def provider_configurations(self) -> ProviderConfigurationRepository:
        """Repository for per-tenant/provider configuration rows, sharing this UoW's session."""
        return self._provider_configurations

    @property
    def ai_model_configurations(self) -> AIModelConfigurationRepository:
        """Repository for per-model configuration overrides, sharing this UoW's session."""
        return self._ai_model_configurations

    @property
    def chat_sessions(self) -> ChatSessionRepository:
        """Repository for chat session records, sharing this UoW's session."""
        return self._chat_sessions

    @property
    def messages(self) -> MessageRepository:
        """Repository for individual chat messages, sharing this UoW's session."""
        return self._messages

    @property
    def prompt_templates(self) -> PromptTemplateRepository:
        """Repository for reusable prompt templates, sharing this UoW's session."""
        return self._prompt_templates

    @property
    def api_keys(self) -> APIKeyRepository:
        """Repository for stored API key records, sharing this UoW's session."""
        return self._api_keys

    @property
    def usage_records(self) -> UsageRecordRepository:
        """Repository for provider usage/billing records, sharing this UoW's session."""
        return self._usage_records

    @property
    def provider_health(self) -> ProviderHealthRepository:
        """Repository for provider health-check history, sharing this UoW's session."""
        return self._provider_health

    async def commit(self) -> None:
        """
        Persist all flushed changes in the current transaction.

        This is the only method that actually writes changes to the
        database — the base class's ``__aexit__`` never calls this
        automatically (see ``BaseUnitOfWork.__aexit__``), so callers must
        explicitly ``await uow.commit()`` once they've made all the changes
        that belong in one atomic unit.

        Returns:
            None.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the underlying database
                rejects the transaction (e.g. a constraint violation), in
                which case the transaction is left in a failed state and the
                caller should let the context manager's exception path
                trigger ``rollback``.

        Example:
            >>> import asyncio
            >>> from unittest.mock import AsyncMock, MagicMock
            >>> from sqlalchemy.ext.asyncio import AsyncSession
            >>> session = MagicMock(spec=AsyncSession)
            >>> session.commit = AsyncMock()
            >>> uow = AsyncUnitOfWork(session)
            >>> asyncio.run(uow.commit())
            >>> session.commit.await_count
            1
        """
        await self._session.commit()

    async def rollback(self) -> None:
        """
        Discard uncommitted changes in the current transaction.

        Called automatically by ``BaseUnitOfWork.__aexit__`` when the
        ``async with`` block exits due to an exception, so that a partially
        completed operation (e.g. a chat session created but its first
        message failed to insert) never leaves inconsistent data committed.

        Returns:
            None.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the underlying session fails
                to roll back (rare, typically indicates the connection is
                already broken).

        Example:
            >>> import asyncio
            >>> from unittest.mock import AsyncMock, MagicMock
            >>> from sqlalchemy.ext.asyncio import AsyncSession
            >>> session = MagicMock(spec=AsyncSession)
            >>> session.rollback = AsyncMock()
            >>> uow = AsyncUnitOfWork(session)
            >>> asyncio.run(uow.rollback())
            >>> session.rollback.await_count
            1
        """
        await self._session.rollback()

    async def close(self) -> None:
        """
        Close the session when configured to manage its lifecycle.

        Guards against double-closing (via ``self._closed``) and against
        closing a session this Unit of Work doesn't own (via
        ``close_session=False``, passed to ``__init__``) — e.g. when a
        request-scoped dependency provider creates the session and is
        responsible for tearing it down itself.

        Returns:
            None. Calling this more than once is safe — the second call is a
            no-op because ``self._closed`` is already ``True``.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the underlying session fails
                to close cleanly.

        Example:
            >>> import asyncio
            >>> from unittest.mock import AsyncMock, MagicMock
            >>> from sqlalchemy.ext.asyncio import AsyncSession
            >>> session = MagicMock(spec=AsyncSession)
            >>> session.close = AsyncMock()
            >>> uow = AsyncUnitOfWork(session)
            >>> asyncio.run(uow.close())
            >>> asyncio.run(uow.close())  # second call is a no-op
            >>> session.close.await_count
            1
        """
        if self._closed or not self._close_session:
            return
        await self._session.close()
        self._closed = True
