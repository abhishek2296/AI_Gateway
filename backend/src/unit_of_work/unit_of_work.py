"""Async SQLAlchemy Unit of Work coordinating all entity repositories."""

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
        return self._providers

    @property
    def ai_models(self) -> AIModelRepository:
        return self._ai_models

    @property
    def provider_configurations(self) -> ProviderConfigurationRepository:
        return self._provider_configurations

    @property
    def ai_model_configurations(self) -> AIModelConfigurationRepository:
        return self._ai_model_configurations

    @property
    def chat_sessions(self) -> ChatSessionRepository:
        return self._chat_sessions

    @property
    def messages(self) -> MessageRepository:
        return self._messages

    @property
    def prompt_templates(self) -> PromptTemplateRepository:
        return self._prompt_templates

    @property
    def api_keys(self) -> APIKeyRepository:
        return self._api_keys

    @property
    def usage_records(self) -> UsageRecordRepository:
        return self._usage_records

    @property
    def provider_health(self) -> ProviderHealthRepository:
        return self._provider_health

    async def commit(self) -> None:
        """Persist all flushed changes in the current transaction."""
        await self._session.commit()

    async def rollback(self) -> None:
        """Discard uncommitted changes in the current transaction."""
        await self._session.rollback()

    async def close(self) -> None:
        """Close the session when configured to manage its lifecycle."""
        if self._closed or not self._close_session:
            return
        await self._session.close()
        self._closed = True
