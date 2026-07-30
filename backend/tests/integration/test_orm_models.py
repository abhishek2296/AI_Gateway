"""
ORM model constraint, relationship, and PostgreSQL type tests.

These tests verify database-level behavior that only shows up against a
real PostgreSQL engine — unique constraints (including partial unique
indexes), foreign key cascade/set-null/restrict behaviors, JSONB and UUID
round-tripping, ``NUMERIC`` precision, and default column values — none of
which a mocked session could meaningfully exercise. Test data is built with
the ``make_*`` factories in ``tests/factories.py``.

Requires a live PostgreSQL instance — see the ``integration`` pytest marker
applied via ``pytestmark`` below.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai_model import AIModel
from src.repositories.provider_repository import ProviderRepository
from tests.helpers import expect_integrity_error
from tests.factories import (
    make_ai_model,
    make_api_key,
    make_chat_session,
    make_message,
    make_prompt_template,
    make_provider,
    make_provider_configuration,
    make_provider_health,
    make_usage_record,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_provider_unique_name(db_session: AsyncSession) -> None:
    """
    ``providers.name`` has a unique index — creating a second provider with
    an already-used name must raise ``IntegrityError``.
    """
    repo = ProviderRepository(db_session)
    await repo.create(make_provider(name="unique-provider"))

    await expect_integrity_error(
        db_session,
        lambda: repo.create(make_provider(name="unique-provider")),
    )


@pytest.mark.asyncio
async def test_ai_model_unique_per_provider(db_session: AsyncSession) -> None:
    """
    ``(provider_id, model_name)`` is uniquely constrained — a provider
    cannot register the same model name twice, though different providers
    may each have a model with that name (not exercised here, but implied
    by the composite key).
    """
    provider_repo = ProviderRepository(db_session)
    provider = await provider_repo.create(make_provider(name="provider-for-model-unique"))

    from src.repositories.ai_model_repository import AIModelRepository

    model_repo = AIModelRepository(db_session)
    await model_repo.create(make_ai_model(provider_id=provider.id, model_name="shared-name"))

    await expect_integrity_error(
        db_session,
        lambda: model_repo.create(make_ai_model(provider_id=provider.id, model_name="shared-name")),
    )


@pytest.mark.asyncio
async def test_prompt_template_unique_name_version(db_session: AsyncSession) -> None:
    """
    ``(name, version)`` is uniquely constrained on ``prompt_templates`` —
    two templates cannot share the same name *and* version number (though
    the same name at different versions is a supported, intentional
    pattern for template revisions).
    """
    from src.repositories.prompt_template_repository import PromptTemplateRepository

    repo = PromptTemplateRepository(db_session)
    await repo.create(make_prompt_template(name="dup-template", version=1))

    await expect_integrity_error(
        db_session,
        lambda: repo.create(make_prompt_template(name="dup-template", version=1)),
    )


@pytest.mark.asyncio
async def test_provider_cascade_deletes_ai_models(db_session: AsyncSession) -> None:
    """
    Deleting a ``Provider`` must cascade-delete its ``AIModel`` rows
    (``ON DELETE CASCADE``), since a model cannot meaningfully exist
    without its owning provider.

    Uses ``db_session.flush()`` (not a full commit) to push the DELETE to
    PostgreSQL and let the database-level cascade fire, then queries via
    the repository to confirm the child row is really gone at the database
    level, not just detached from the ORM session's identity map.
    """
    provider_repo = ProviderRepository(db_session)
    provider = await provider_repo.create(make_provider(name="cascade-provider"))

    from src.repositories.ai_model_repository import AIModelRepository

    model_repo = AIModelRepository(db_session)
    model = await model_repo.create(make_ai_model(provider_id=provider.id))

    await provider_repo.delete(provider)
    await db_session.flush()

    assert await model_repo.count(AIModel.id == model.id) == 0


@pytest.mark.asyncio
async def test_chat_session_cascade_deletes_messages(db_session: AsyncSession) -> None:
    """
    Deleting a ``ChatSession`` must cascade-delete its ``Message`` rows
    (``ON DELETE CASCADE``), consistent with the provider/model cascade
    above — conversation messages have no meaning without their session.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="msg-cascade-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.chat_session_repository import ChatSessionRepository
    from src.repositories.message_repository import MessageRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    session = await ChatSessionRepository(db_session).create(
        make_chat_session(provider_id=provider.id, ai_model_id=model.id),
    )
    message = await MessageRepository(db_session).create(
        make_message(session_id=session.id, content="cascade me"),
    )

    from src.models.message import Message

    await ChatSessionRepository(db_session).delete(session)
    await db_session.flush()

    assert await MessageRepository(db_session).count(Message.id == message.id) == 0


@pytest.mark.asyncio
async def test_usage_record_set_null_on_session_delete(db_session: AsyncSession) -> None:
    """
    Deleting a ``ChatSession`` that has associated ``UsageRecord`` rows
    must set their ``chat_session_id`` to ``NULL`` (``ON DELETE SET NULL``)
    rather than deleting the usage rows — billing/telemetry history must
    outlive the conversational session it originated from.

    ``db_session.refresh(usage)`` re-fetches the row's current state from
    the database after the session deletion, since the ORM's in-memory
    object wouldn't otherwise know the database-level ``SET NULL`` trigger
    fired.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="set-null-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.chat_session_repository import ChatSessionRepository
    from src.repositories.usage_record_repository import UsageRecordRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    session = await ChatSessionRepository(db_session).create(
        make_chat_session(provider_id=provider.id, ai_model_id=model.id),
    )
    usage_repo = UsageRecordRepository(db_session)
    usage = await usage_repo.create(
        make_usage_record(
            provider_id=provider.id,
            ai_model_id=model.id,
            chat_session_id=session.id,
            request_id="set-null-req",
        ),
    )

    await ChatSessionRepository(db_session).delete(session)
    await db_session.refresh(usage)

    assert usage.chat_session_id is None


@pytest.mark.asyncio
async def test_usage_record_restricts_provider_delete(db_session: AsyncSession) -> None:
    """
    Deleting a ``Provider`` that still has ``UsageRecord`` rows referencing
    it must fail with ``IntegrityError`` (``ON DELETE RESTRICT``) — unlike
    the cascade behavior for ``AIModel``/``Message``, billing history must
    never be silently deleted just because its provider is removed.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="restrict-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.usage_record_repository import UsageRecordRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    await UsageRecordRepository(db_session).create(
        make_usage_record(provider_id=provider.id, ai_model_id=model.id, request_id="restrict-req"),
    )

    await expect_integrity_error(
        db_session,
        lambda: ProviderRepository(db_session).delete(provider),
    )


@pytest.mark.asyncio
async def test_jsonb_metadata_roundtrip(db_session: AsyncSession) -> None:
    """
    A nested ``dict``/``list`` structure written to a ``JSONB`` column
    (``ChatSession.extra_metadata``) must be read back byte-for-byte
    equivalent after a round trip through PostgreSQL.

    Confirms SQLAlchemy's JSONB type handling correctly (de)serializes
    complex structures (a dict with a nested list value here), not just
    flat key/value pairs.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="jsonb-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.chat_session_repository import ChatSessionRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    metadata = {"tenant": "acme", "tags": ["alpha", "beta"]}
    session = await ChatSessionRepository(db_session).create(
        make_chat_session(
            provider_id=provider.id,
            ai_model_id=model.id,
            extra_metadata=metadata,
        ),
    )

    loaded = await ChatSessionRepository(db_session).get_by_id(session.id)
    assert loaded is not None
    assert loaded.extra_metadata == metadata


@pytest.mark.asyncio
async def test_uuid_session_roundtrip(db_session: AsyncSession) -> None:
    """
    A Python ``uuid.UUID`` written to ``ChatSession.session_uuid`` (a
    PostgreSQL native ``uuid`` column) must be usable to look the row back
    up afterward via ``get_by_uuid``.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="uuid-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.chat_session_repository import ChatSessionRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    session_uuid = uuid.uuid4()
    created = await ChatSessionRepository(db_session).create(
        make_chat_session(
            provider_id=provider.id,
            ai_model_id=model.id,
            session_uuid=session_uuid,
        ),
    )

    loaded = await ChatSessionRepository(db_session).get_by_uuid(session_uuid)
    assert loaded is not None
    assert loaded.id == created.id


@pytest.mark.asyncio
async def test_numeric_cost_precision(db_session: AsyncSession) -> None:
    """
    A high-precision ``Decimal`` cost value (6 decimal places, matching the
    column's ``NUMERIC(12, 6)`` definition) must round-trip through
    PostgreSQL exactly, with no floating-point rounding error.

    Using ``Decimal`` (not ``float``) throughout is essential for billing
    calculations — this test guards against a regression where the column
    type or ORM mapping silently loses precision.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="numeric-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.usage_record_repository import UsageRecordRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    cost = Decimal("123456.654321")
    usage = await UsageRecordRepository(db_session).create(
        make_usage_record(
            provider_id=provider.id,
            ai_model_id=model.id,
            request_id="numeric-req",
            estimated_cost=cost,
        ),
    )

    loaded = await UsageRecordRepository(db_session).get_by_id(usage.id)
    assert loaded is not None
    assert loaded.estimated_cost == cost


@pytest.mark.asyncio
async def test_timestamps_populated(db_session: AsyncSession) -> None:
    """
    ``created_at``/``updated_at`` columns must be automatically populated by
    their server-side ``now()`` defaults, without the application needing
    to set them explicitly.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="timestamp-provider"))

    from src.repositories.provider_configuration_repository import ProviderConfigurationRepository

    configuration = await ProviderConfigurationRepository(db_session).create(
        make_provider_configuration(provider_id=provider.id),
    )

    assert configuration.created_at is not None
    assert configuration.updated_at is not None


@pytest.mark.asyncio
async def test_nullable_fields(db_session: AsyncSession) -> None:
    """
    Optional ``Provider`` columns (``description``, ``base_url``) must
    accept and persist an explicit ``None`` value without raising, since
    the schema declares them nullable.
    """
    provider = await ProviderRepository(db_session).create(
        make_provider(name="nullable-provider", description=None, base_url=None),
    )
    assert provider.description is None
    assert provider.base_url is None


@pytest.mark.asyncio
async def test_provider_relationships_load(db_session: AsyncSession) -> None:
    """
    A provider's related rows across four different child tables
    (configuration, model, API key, health check) must each be independently
    queryable by their respective repositories once created.

    This is a broad smoke test confirming all of a provider's one-to-one and
    one-to-many relationships are wired correctly end-to-end, rather than
    testing each relationship type in isolation.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="rel-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.api_key_repository import APIKeyRepository
    from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
    from src.repositories.provider_health_repository import ProviderHealthRepository

    await ProviderConfigurationRepository(db_session).create(
        make_provider_configuration(provider_id=provider.id),
    )
    await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    await APIKeyRepository(db_session).create(make_api_key(provider_id=provider.id))
    await ProviderHealthRepository(db_session).create(make_provider_health(provider_id=provider.id))

    assert await ProviderConfigurationRepository(db_session).get_by_provider_id(provider.id) is not None
    assert await AIModelRepository(db_session).count(AIModel.provider_id == provider.id) == 1
    assert len(await APIKeyRepository(db_session).list_active_for_provider(provider.id)) == 1
    assert len(await ProviderHealthRepository(db_session).list_for_provider(provider.id)) == 1


@pytest.mark.asyncio
async def test_default_boolean_values(db_session: AsyncSession) -> None:
    """
    Boolean columns must correctly reflect an explicitly passed value
    (``is_active=True``) alongside a column relying on its server-side
    default (``is_local``, expected to default to ``False``).
    """
    provider = await ProviderRepository(db_session).create(
        make_provider(name="defaults-provider", is_active=True),
    )
    assert provider.is_active is True
    assert provider.is_local is False


@pytest.mark.asyncio
async def test_usage_record_unique_request_id(db_session: AsyncSession) -> None:
    """
    ``usage_records.request_id`` is uniquely constrained
    (``uq_usage_records_request_id``) — recording usage twice under the
    same request id must raise ``IntegrityError``, preventing accidental
    double-billing for a single logical request.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="usage-unique-provider"))
    from src.repositories.ai_model_repository import AIModelRepository
    from src.repositories.usage_record_repository import UsageRecordRepository

    model = await AIModelRepository(db_session).create(make_ai_model(provider_id=provider.id))
    usage_repo = UsageRecordRepository(db_session)
    await usage_repo.create(
        make_usage_record(
            provider_id=provider.id,
            ai_model_id=model.id,
            request_id="unique-req-001",
        ),
    )

    await expect_integrity_error(
        db_session,
        lambda: usage_repo.create(
            make_usage_record(
                provider_id=provider.id,
                ai_model_id=model.id,
                request_id="unique-req-001",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_one_default_ai_model_per_provider(db_session: AsyncSession) -> None:
    """
    A provider may have at most one ``AIModel`` flagged ``is_default=True``,
    enforced by the partial unique index
    ``uq_ai_models_one_default_per_provider`` (unique on ``provider_id``
    filtered to rows where ``is_default IS TRUE``).

    Non-default models can coexist freely (as shown by
    ``secondary-model`` here) — the constraint only kicks in once a *second*
    default is attempted for the same provider.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="default-model-provider"))
    from src.repositories.ai_model_repository import AIModelRepository

    model_repo = AIModelRepository(db_session)
    await model_repo.create(
        make_ai_model(provider_id=provider.id, model_name="default-model", is_default=True),
    )
    await model_repo.create(
        make_ai_model(provider_id=provider.id, model_name="secondary-model", is_default=False),
    )

    await expect_integrity_error(
        db_session,
        lambda: model_repo.create(
            make_ai_model(provider_id=provider.id, model_name="other-default", is_default=True),
        ),
    )


@pytest.mark.asyncio
async def test_one_default_api_key_per_provider(db_session: AsyncSession) -> None:
    """
    A provider may have at most one ``APIKey`` flagged ``is_default=True``,
    enforced by the partial unique index
    ``uq_api_keys_one_default_per_provider`` — the API key analog of
    ``test_one_default_ai_model_per_provider``.
    """
    provider = await ProviderRepository(db_session).create(make_provider(name="default-key-provider"))
    from src.repositories.api_key_repository import APIKeyRepository

    key_repo = APIKeyRepository(db_session)
    await key_repo.create(
        make_api_key(provider_id=provider.id, name="primary", is_default=True),
    )
    await key_repo.create(
        make_api_key(provider_id=provider.id, name="secondary", is_default=False),
    )

    await expect_integrity_error(
        db_session,
        lambda: key_repo.create(
            make_api_key(provider_id=provider.id, name="other-default", is_default=True),
        ),
    )
