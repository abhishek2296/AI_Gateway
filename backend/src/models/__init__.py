"""
ORM model foundation — declarative base, mixins, and domain entities.

This package is the persistence layer's schema definition: every table the
AI Gateway writes to or reads from is declared here as a SQLAlchemy 2.x
``Mapped``/``mapped_column`` entity. Nothing in ``src/models/`` talks HTTP,
calls a provider SDK, or contains business logic — per the layer boundaries
in ``01-engineering-principles.mdc``, models are pure schema/ORM definitions
consumed by the ``services/`` layer.

Domain shape at a glance:

- :class:`~src.models.provider.Provider` — an LLM backend (Ollama, OpenAI, ...).
- :class:`~src.models.provider_configuration.ProviderConfiguration` — 1:1
  connection settings for a provider.
- :class:`~src.models.provider_health.ProviderHealth` — historical health
  check snapshots for a provider.
- :class:`~src.models.api_key.APIKey` — credential *references* (env var
  names, never secrets) for a provider.
- :class:`~src.models.ai_model.AIModel` — a specific model under a provider
  (e.g. ``gpt-4o``).
- :class:`~src.models.ai_model_configuration.AIModelConfiguration` — 1:1
  default generation settings for a model.
- :class:`~src.models.chat_session.ChatSession` — a conversation bound to a
  provider + model.
- :class:`~src.models.message.Message` — one turn within a chat session.
- :class:`~src.models.usage_record.UsageRecord` — immutable per-request
  usage/billing audit row.
- :class:`~src.models.prompt_template.PromptTemplate` — reusable,
  versioned prompt templates (not yet linked to sessions).

This module re-exports every entity (plus :class:`~src.models.base.Base` and
:class:`~src.models.mixins.TimestampMixin`) so callers can do
``from src.models import Provider`` instead of reaching into each
submodule individually. Importing everything here also guarantees all
model classes are registered on ``Base.metadata`` before Alembic
autogeneration or ``Base.metadata.create_all()`` runs.
"""

from src.models.ai_model import AIModel
from src.models.ai_model_configuration import AIModelConfiguration
from src.models.api_key import APIKey
from src.models.base import Base
from src.models.chat_session import ChatSession
from src.models.message import Message
from src.models.mixins import TimestampMixin
from src.models.prompt_template import PromptTemplate
from src.models.provider import Provider
from src.models.provider_configuration import ProviderConfiguration
from src.models.provider_health import ProviderHealth
from src.models.usage_record import UsageRecord

__all__ = [
    "Base",
    "TimestampMixin",
    "Provider",
    "ProviderConfiguration",
    "ProviderHealth",
    "AIModel",
    "AIModelConfiguration",
    "APIKey",
    "ChatSession",
    "Message",
    "PromptTemplate",
    "UsageRecord",
]
