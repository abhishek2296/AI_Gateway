"""
Async repository layer for AI Gateway persistence.

This package implements the Repository pattern (see
``01-engineering-principles.mdc``'s layer boundaries) on top of SQLAlchemy
2.x's async ORM: every persisted entity in ``src/models/`` has a matching
repository here that owns all reads/writes for that table. Services and the
future Unit of Work layer depend on these repositories rather than importing
SQLAlchemy directly, keeping query logic centralized and testable.

Every repository extends :class:`~src.repositories.base.BaseRepository` for
generic CRUD (create/get/list/update/delete/count/exists) and adds only the
query methods specific to its entity — e.g.
:meth:`~src.repositories.provider_repository.ProviderRepository.get_by_name`.
Re-exporting every repository class here lets callers write
``from src.repositories import ProviderRepository`` instead of reaching into
each submodule individually.
"""

from src.repositories.ai_model_configuration_repository import AIModelConfigurationRepository
from src.repositories.ai_model_repository import AIModelRepository
from src.repositories.api_key_repository import APIKeyRepository
from src.repositories.base import BaseRepository
from src.repositories.chat_session_repository import ChatSessionRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.prompt_template_repository import PromptTemplateRepository
from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
from src.repositories.provider_health_repository import ProviderHealthRepository
from src.repositories.provider_repository import ProviderRepository
from src.repositories.usage_record_repository import UsageRecordRepository

__all__ = [
    "BaseRepository",
    "ProviderRepository",
    "AIModelRepository",
    "ProviderConfigurationRepository",
    "AIModelConfigurationRepository",
    "ChatSessionRepository",
    "MessageRepository",
    "PromptTemplateRepository",
    "APIKeyRepository",
    "UsageRecordRepository",
    "ProviderHealthRepository",
]
