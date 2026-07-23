"""ORM model constraint, relationship, and PostgreSQL type tests."""

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
    repo = ProviderRepository(db_session)
    await repo.create(make_provider(name="unique-provider"))

    await expect_integrity_error(
        db_session,
        lambda: repo.create(make_provider(name="unique-provider")),
    )


@pytest.mark.asyncio
async def test_ai_model_unique_per_provider(db_session: AsyncSession) -> None:
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
    from src.repositories.prompt_template_repository import PromptTemplateRepository

    repo = PromptTemplateRepository(db_session)
    await repo.create(make_prompt_template(name="dup-template", version=1))

    await expect_integrity_error(
        db_session,
        lambda: repo.create(make_prompt_template(name="dup-template", version=1)),
    )


@pytest.mark.asyncio
async def test_provider_cascade_deletes_ai_models(db_session: AsyncSession) -> None:
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
    provider = await ProviderRepository(db_session).create(make_provider(name="timestamp-provider"))

    from src.repositories.provider_configuration_repository import ProviderConfigurationRepository

    configuration = await ProviderConfigurationRepository(db_session).create(
        make_provider_configuration(provider_id=provider.id),
    )

    assert configuration.created_at is not None
    assert configuration.updated_at is not None


@pytest.mark.asyncio
async def test_nullable_fields(db_session: AsyncSession) -> None:
    provider = await ProviderRepository(db_session).create(
        make_provider(name="nullable-provider", description=None, base_url=None),
    )
    assert provider.description is None
    assert provider.base_url is None


@pytest.mark.asyncio
async def test_provider_relationships_load(db_session: AsyncSession) -> None:
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
    provider = await ProviderRepository(db_session).create(
        make_provider(name="defaults-provider", is_active=True),
    )
    assert provider.is_active is True
    assert provider.is_local is False


@pytest.mark.asyncio
async def test_usage_record_unique_request_id(db_session: AsyncSession) -> None:
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
