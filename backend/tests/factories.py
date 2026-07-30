"""
Shared entity builders for persistence integration tests.

Each ``make_*`` function builds one ORM model instance (unsaved — the caller
is responsible for passing it to a repository's ``create()``) with
realistic, valid default field values. Callers override only the fields
relevant to the behavior under test via ``**overrides``, keeping integration
tests focused on what they're actually verifying instead of restating every
required column on every entity.

Foreign keys (``provider_id``, ``ai_model_id``, etc.) are always required
keyword-only arguments rather than defaults, since a valid parent row must
already exist in the test database before a child row can be created.
"""

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
    """
    Build an unsaved ``Provider`` with realistic defaults (mirrors a local
    Ollama deployment).

    Args:
        name: Unique provider name/identifier. Tests should pass a distinct
            value (often with a random suffix) whenever uniqueness
            constraints are relevant, since ``name`` has a unique index.
        **overrides: Any other ``Provider`` column to override, e.g.
            ``is_active=False`` or ``base_url=None``.

    Returns:
        A ``Provider`` instance, not yet persisted. Pass it to
        ``ProviderRepository(session).create(...)`` to save it.

    Example:
        >>> provider = make_provider(name="test-provider", is_active=False)
        >>> provider.is_active
        False
    """
    defaults = {
        "display_name": "Ollama",
        "provider_type": "ollama",
        "is_active": True,
    }
    defaults.update(overrides)
    return Provider(name=name, **defaults)


def make_ai_model(*, provider_id: int, model_name: str = "qwen3:8b", **overrides) -> AIModel:
    """
    Build an unsaved ``AIModel`` belonging to an existing provider.

    Args:
        provider_id: Primary key of an already-persisted ``Provider`` row
            this model belongs to.
        model_name: Model identifier, unique per provider (see
            ``uq_ai_models_provider_id_model_name``). Tests exercising that
            constraint should pass a duplicate value deliberately.
        **overrides: Any other ``AIModel`` column to override, e.g.
            ``is_default=True`` or ``is_active=False``.

    Returns:
        An ``AIModel`` instance, not yet persisted.

    Example:
        >>> model = make_ai_model(provider_id=1, model_name="qwen3:8b", is_default=True)
        >>> model.is_default
        True
    """
    defaults = {
        "display_name": "Qwen3 8B",
        "is_active": True,
        "is_default": False,
    }
    defaults.update(overrides)
    return AIModel(provider_id=provider_id, model_name=model_name, **defaults)


def make_provider_configuration(*, provider_id: int, **overrides) -> ProviderConfiguration:
    """
    Build an unsaved ``ProviderConfiguration`` (connection settings) for an
    existing provider.

    A provider may have at most one configuration row (enforced by
    ``uq_provider_configurations_provider_id``), so tests should not call
    this twice for the same ``provider_id`` without expecting an
    ``IntegrityError``.

    Args:
        provider_id: Primary key of an already-persisted ``Provider`` row.
        **overrides: Any other ``ProviderConfiguration`` column to
            override, e.g. ``endpoint="http://custom:11434"``.

    Returns:
        A ``ProviderConfiguration`` instance, not yet persisted.

    Example:
        >>> config = make_provider_configuration(provider_id=1, timeout_seconds=30)
        >>> config.timeout_seconds
        30
    """
    defaults = {
        "api_key_env": "OLLAMA_API_KEY",
        "timeout_seconds": 60,
        "is_active": True,
    }
    defaults.update(overrides)
    return ProviderConfiguration(provider_id=provider_id, **defaults)


def make_ai_model_configuration(*, ai_model_id: int, **overrides) -> AIModelConfiguration:
    """
    Build an unsaved ``AIModelConfiguration`` (generation parameters) for an
    existing model.

    An AI model may have at most one configuration row (enforced by
    ``uq_ai_model_configurations_ai_model_id``).

    Args:
        ai_model_id: Primary key of an already-persisted ``AIModel`` row.
        **overrides: Any other ``AIModelConfiguration`` column to override,
            e.g. ``temperature=0.2``.

    Returns:
        An ``AIModelConfiguration`` instance, not yet persisted.

    Example:
        >>> config = make_ai_model_configuration(ai_model_id=1, max_tokens=2048)
        >>> config.max_tokens
        2048
    """
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
    """
    Build an unsaved ``ChatSession`` linked to an existing provider and model.

    Args:
        provider_id: Primary key of an already-persisted ``Provider`` row.
        ai_model_id: Primary key of an already-persisted ``AIModel`` row.
        session_uuid: Public-facing session identifier. If omitted, a fresh
            random ``uuid4`` is generated — pass an explicit value when a
            test needs to look the session up by a known UUID afterwards.
        **overrides: Any other ``ChatSession`` column to override, e.g.
            ``is_archived=True`` or ``extra_metadata={...}``.

    Returns:
        A ``ChatSession`` instance, not yet persisted.

    Example:
        >>> session_id = uuid.uuid4()
        >>> session = make_chat_session(provider_id=1, ai_model_id=1, session_uuid=session_id)
        >>> session.session_uuid == session_id
        True
    """
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
    """
    Build an unsaved ``Message`` belonging to an existing chat session.

    Args:
        session_id: Primary key of an already-persisted ``ChatSession`` row.
        role: Message role, e.g. ``"user"``, ``"assistant"``, or
            ``"system"``.
        content: Message text body.
        **overrides: Any other ``Message`` column to override, e.g.
            ``token_count=42``.

    Returns:
        A ``Message`` instance, not yet persisted.

    Example:
        >>> message = make_message(session_id=1, role="assistant", content="Hi there")
        >>> message.role
        'assistant'
    """
    defaults = {
        "extra_metadata": {"lang": "en"},
        "token_count": 5,
    }
    defaults.update(overrides)
    return Message(session_id=session_id, role=role, content=content, **defaults)


def make_prompt_template(*, name: str = "greeting", version: int = 1, **overrides) -> PromptTemplate:
    """
    Build an unsaved ``PromptTemplate``.

    The ``(name, version)`` pair is unique (see
    ``uq_prompt_templates_name_version``), so tests exercising that
    constraint should intentionally reuse both values.

    Args:
        name: Template name. Combined with ``version`` for uniqueness.
        version: Template version number, starting at ``1``.
        **overrides: Any other ``PromptTemplate`` column to override, e.g.
            ``is_active=False``.

    Returns:
        A ``PromptTemplate`` instance, not yet persisted.

    Example:
        >>> template = make_prompt_template(name="greeting", version=2)
        >>> template.version
        2
    """
    defaults = {
        "template": "Hello {{name}}",
        "variables": {"name": "World"},
        "is_active": True,
    }
    defaults.update(overrides)
    return PromptTemplate(name=name, version=version, **defaults)


def make_api_key(*, provider_id: int, name: str = "default", **overrides) -> APIKey:
    """
    Build an unsaved ``APIKey`` reference (never a real secret — only the
    identifying metadata and the name of the env var holding the secret)
    for an existing provider.

    Args:
        provider_id: Primary key of an already-persisted ``Provider`` row.
        name: Key name, unique per provider (see
            ``uq_api_keys_provider_id_name``).
        **overrides: Any other ``APIKey`` column to override, e.g.
            ``is_default=True`` or ``expires_at=...``.

    Returns:
        An ``APIKey`` instance, not yet persisted.

    Example:
        >>> key = make_api_key(provider_id=1, name="primary", is_default=True)
        >>> key.is_default
        True
    """
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
    """
    Build an unsaved ``UsageRecord`` (billing/telemetry row) for an existing
    provider and model.

    Args:
        provider_id: Primary key of an already-persisted ``Provider`` row.
            Deleting a provider with usage records is restricted (see
            ``FK ... ON DELETE RESTRICT`` in the schema) to preserve billing
            history.
        ai_model_id: Primary key of an already-persisted ``AIModel`` row.
        request_id: Unique request identifier (see
            ``uq_usage_records_request_id``); tests exercising that
            constraint should pass a duplicate value deliberately.
        chat_session_id: Primary key of an optional owning ``ChatSession``.
            ``None`` represents usage not tied to any interactive session.
            Deleting the session sets this to ``NULL`` rather than deleting
            the usage row (``ON DELETE SET NULL``), since usage history must
            outlive the session.
        **overrides: Any other ``UsageRecord`` column to override, e.g.
            ``estimated_cost=Decimal("1.23")``.

    Returns:
        A ``UsageRecord`` instance, not yet persisted.

    Example:
        >>> usage = make_usage_record(provider_id=1, ai_model_id=1, request_id="req-abc")
        >>> usage.status
        'success'
    """
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
    """
    Build an unsaved ``ProviderHealth`` snapshot (a point-in-time health
    check result) for an existing provider.

    Multiple health rows may exist per provider over time (unlike
    ``ProviderConfiguration``, this is not a one-to-one relationship) —
    each call represents one health check event.

    Args:
        provider_id: Primary key of an already-persisted ``Provider`` row.
        status: Health status string, e.g. ``"healthy"`` or ``"unhealthy"``.
        **overrides: Any other ``ProviderHealth`` column to override, e.g.
            ``checked_at=...`` to control chronological ordering in tests.

    Returns:
        A ``ProviderHealth`` instance, not yet persisted.

    Example:
        >>> health = make_provider_health(provider_id=1, status="unhealthy")
        >>> health.status
        'unhealthy'
    """
    defaults = {
        "latency_ms": 45,
        "health_score": 0.99,
        "checked_at": datetime.now(tz=UTC),
    }
    defaults.update(overrides)
    return ProviderHealth(provider_id=provider_id, status=status, **defaults)
