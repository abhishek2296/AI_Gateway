"""Shared entity builders for persistence integration tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from src.models.ai_model import AIModel
from src.models.ai_model_configuration import AIModelConfiguration
from src.models.api_key import APIKey
from src.models.chat_session import ChatSession
from src.models.message import Message
from src.models.prompt_template import PromptTemplate
from src.models.provider import Provider
from src.models.provider_configuration import ProviderConfiguration
from src.models.provider_health import ProviderHealth
from src.models.usage_record import UsageRecord


def make_provider(*, name: str = "ollama", **overrides) -> Provider:
    defaults = {
        "display_name": "Ollama",
        "provider_type": "ollama",
        "is_active": True,
    }
    defaults.update(overrides)
    return Provider(name=name, **defaults)


def make_ai_model(*, provider_id: int, model_name: str = "qwen3:8b", **overrides) -> AIModel:
    defaults = {
        "display_name": "Qwen3 8B",
        "is_active": True,
        "is_default": False,
    }
    defaults.update(overrides)
    return AIModel(provider_id=provider_id, model_name=model_name, **defaults)


def make_provider_configuration(*, provider_id: int, **overrides) -> ProviderConfiguration:
    defaults = {
        "api_key_env": "OLLAMA_API_KEY",
        "timeout_seconds": 60,
        "is_active": True,
    }
    defaults.update(overrides)
    return ProviderConfiguration(provider_id=provider_id, **defaults)


def make_ai_model_configuration(*, ai_model_id: int, **overrides) -> AIModelConfiguration:
    defaults = {
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    defaults.update(overrides)
    return AIModelConfiguration(ai_model_id=ai_model_id, **defaults)


def make_chat_session(
    *,
    provider_id: int,
    ai_model_id: int,
    session_uuid: uuid.UUID | None = None,
    **overrides,
) -> ChatSession:
    defaults = {
        "session_uuid": session_uuid or uuid.uuid4(),
        "title": "Test session",
        "extra_metadata": {"source": "test"},
        "is_archived": False,
    }
    defaults.update(overrides)
    return ChatSession(
        provider_id=provider_id,
        ai_model_id=ai_model_id,
        **defaults,
    )


def make_message(*, session_id: int, role: str = "user", content: str = "Hello", **overrides) -> Message:
    defaults = {
        "extra_metadata": {"lang": "en"},
        "token_count": 5,
    }
    defaults.update(overrides)
    return Message(session_id=session_id, role=role, content=content, **defaults)


def make_prompt_template(*, name: str = "greeting", version: int = 1, **overrides) -> PromptTemplate:
    defaults = {
        "template": "Hello {{name}}",
        "variables": {"name": "World"},
        "is_active": True,
    }
    defaults.update(overrides)
    return PromptTemplate(name=name, version=version, **defaults)


def make_api_key(*, provider_id: int, name: str = "default", **overrides) -> APIKey:
    defaults = {
        "key_identifier": "test-key",
        "api_key_env": "TEST_API_KEY",
        "is_default": False,
        "is_active": True,
    }
    defaults.update(overrides)
    return APIKey(provider_id=provider_id, name=name, **defaults)


def make_usage_record(
    *,
    provider_id: int,
    ai_model_id: int,
    request_id: str = "req-test-001",
    chat_session_id: int | None = None,
    **overrides,
) -> UsageRecord:
    defaults = {
        "chat_session_id": chat_session_id,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "estimated_cost": Decimal("0.001500"),
        "latency_ms": 120,
        "status": "success",
        "request_timestamp": datetime.now(tz=UTC),
    }
    defaults.update(overrides)
    return UsageRecord(
        provider_id=provider_id,
        ai_model_id=ai_model_id,
        request_id=request_id,
        **defaults,
    )


def make_provider_health(*, provider_id: int, status: str = "healthy", **overrides) -> ProviderHealth:
    defaults = {
        "latency_ms": 45,
        "health_score": 0.99,
        "checked_at": datetime.now(tz=UTC),
    }
    defaults.update(overrides)
    return ProviderHealth(provider_id=provider_id, status=status, **defaults)
