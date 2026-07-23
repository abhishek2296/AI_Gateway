"""Integration tests for all persistence repositories."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.provider import Provider
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
from tests.factories import (
    make_ai_model,
    make_ai_model_configuration,
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


async def seed_provider_and_model(session: AsyncSession) -> tuple:
    provider = await ProviderRepository(session).create(make_provider(name=f"repo-{uuid.uuid4().hex[:8]}"))
    model = await AIModelRepository(session).create(
        make_ai_model(provider_id=provider.id, model_name="repo-model"),
    )
    return provider, model


class TestProviderRepository:
    @pytest.mark.asyncio
    async def test_crud(self, db_session: AsyncSession) -> None:
        repo = ProviderRepository(db_session)
        created = await repo.create(make_provider(name="crud-provider"))
        assert created.id is not None

        loaded = await repo.get_by_id(created.id)
        assert loaded is not None
        assert loaded.name == "crud-provider"

        updated = await repo.update(loaded, display_name="Updated")
        assert updated.display_name == "Updated"

        assert await repo.exists(Provider.name == "crud-provider")
        assert await repo.count(Provider.name == "crud-provider") == 1

        assert await repo.delete_by_id(created.id) is True
        assert await repo.get_by_id(created.id) is None
        assert await repo.delete_by_id(created.id) is False

    @pytest.mark.asyncio
    async def test_get_by_name(self, db_session: AsyncSession) -> None:
        repo = ProviderRepository(db_session)
        await repo.create(make_provider(name="named-provider"))
        found = await repo.get_by_name("named-provider")
        assert found is not None

    @pytest.mark.asyncio
    async def test_list_active(self, db_session: AsyncSession) -> None:
        repo = ProviderRepository(db_session)
        await repo.create(make_provider(name="active-provider", is_active=True))
        await repo.create(make_provider(name="inactive-provider", is_active=False))
        active = await repo.list_active()
        names = {p.name for p in active}
        assert "active-provider" in names
        assert "inactive-provider" not in names


class TestAIModelRepository:
    @pytest.mark.asyncio
    async def test_crud_and_queries(self, db_session: AsyncSession) -> None:
        provider, _ = await seed_provider_and_model(db_session)
        repo = AIModelRepository(db_session)

        default = await repo.create(
            make_ai_model(provider_id=provider.id, model_name="default-model", is_default=True),
        )
        disabled = await repo.create(
            make_ai_model(provider_id=provider.id, model_name="disabled-model", is_active=False),
        )

        assert await repo.get_by_provider_and_name(provider.id, "default-model") is not None
        enabled = await repo.list_enabled_models(provider.id)
        enabled_names = {m.model_name for m in enabled}
        assert "default-model" in enabled_names
        assert "disabled-model" not in enabled_names
        assert await repo.get_default_model(provider.id) is not None
        assert (await repo.get_default_model(provider.id)).id == default.id

        await repo.delete(default)
        await repo.delete(disabled)


class TestProviderConfigurationRepository:
    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        provider = await ProviderRepository(db_session).create(make_provider(name="config-provider"))
        repo = ProviderConfigurationRepository(db_session)
        config = await repo.create(make_provider_configuration(provider_id=provider.id))

        assert await repo.get_by_provider_id(provider.id) is not None
        active = await repo.list_active()
        assert any(item.id == config.id for item in active)


class TestAIModelConfigurationRepository:
    @pytest.mark.asyncio
    async def test_get_by_ai_model_id(self, db_session: AsyncSession) -> None:
        provider, model = await seed_provider_and_model(db_session)
        repo = AIModelConfigurationRepository(db_session)
        config = await repo.create(make_ai_model_configuration(ai_model_id=model.id))

        loaded = await repo.get_by_ai_model_id(model.id)
        assert loaded is not None
        assert loaded.id == config.id


class TestChatSessionRepository:
    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        provider, model = await seed_provider_and_model(db_session)
        repo = ChatSessionRepository(db_session)
        session_uuid = uuid.uuid4()
        session = await repo.create(
            make_chat_session(provider_id=provider.id, ai_model_id=model.id, session_uuid=session_uuid),
        )
        archived = await repo.create(
            make_chat_session(
                provider_id=provider.id,
                ai_model_id=model.id,
                is_archived=True,
            ),
        )

        assert await repo.get_by_uuid(session_uuid) is not None
        recent = await repo.list_recent_sessions()
        recent_ids = {item.id for item in recent}
        assert session.id in recent_ids
        assert archived.id not in recent_ids

        archived_result = await repo.archive_session(session)
        assert archived_result.is_archived is True


class TestMessageRepository:
    @pytest.mark.asyncio
    async def test_list_and_count(self, db_session: AsyncSession) -> None:
        provider, model = await seed_provider_and_model(db_session)
        session = await ChatSessionRepository(db_session).create(
            make_chat_session(provider_id=provider.id, ai_model_id=model.id),
        )
        repo = MessageRepository(db_session)
        first = await repo.create(make_message(session_id=session.id, content="one"))
        second = await repo.create(make_message(session_id=session.id, content="two"))

        messages = await repo.list_messages(session.id)
        assert [m.id for m in messages] == [first.id, second.id]
        assert await repo.count_for_session(session.id) == 2


class TestPromptTemplateRepository:
    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        repo = PromptTemplateRepository(db_session)
        active = await repo.create(make_prompt_template(name="active-template", version=1))
        await repo.create(
            make_prompt_template(name="inactive-template", version=1, is_active=False),
        )

        assert await repo.get_by_name_and_version("active-template", 1) is not None
        active_rows = await repo.list_active(category=None)
        names = {row.name for row in active_rows}
        assert "active-template" in names
        assert "inactive-template" not in names
        assert active.id is not None


class TestAPIKeyRepository:
    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        provider = await ProviderRepository(db_session).create(make_provider(name="api-key-provider"))
        repo = APIKeyRepository(db_session)
        default = await repo.create(
            make_api_key(provider_id=provider.id, name="primary", is_default=True),
        )
        await repo.create(make_api_key(provider_id=provider.id, name="secondary", is_active=False))

        assert await repo.get_by_provider_and_name(provider.id, "primary") is not None
        assert (await repo.get_default_for_provider(provider.id)).id == default.id
        active = await repo.list_active_for_provider(provider.id)
        assert len(active) == 1


class TestUsageRecordRepository:
    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        provider, model = await seed_provider_and_model(db_session)
        repo = UsageRecordRepository(db_session)
        now = datetime.now(tz=UTC)
        record = await repo.create(
            make_usage_record(
                provider_id=provider.id,
                ai_model_id=model.id,
                request_id="usage-req-1",
                request_timestamp=now,
            ),
        )

        assert await repo.get_by_request_id("usage-req-1") is not None
        between = await repo.usage_between_dates(now - timedelta(hours=1), now + timedelta(hours=1))
        assert any(item.id == record.id for item in between)
        by_provider = await repo.usage_by_provider(provider.id)
        assert any(item.id == record.id for item in by_provider)


class TestProviderHealthRepository:
    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        provider = await ProviderRepository(db_session).create(make_provider(name="health-provider"))
        repo = ProviderHealthRepository(db_session)
        healthy = await repo.create(make_provider_health(provider_id=provider.id, status="healthy"))
        failed = await repo.create(
            make_provider_health(
                provider_id=provider.id,
                status="unhealthy",
                checked_at=datetime.now(tz=UTC) + timedelta(minutes=1),
            ),
        )

        latest = await repo.latest_health(provider.id)
        assert latest is not None
        assert latest.id == failed.id

        failures = await repo.failed_checks(provider.id)
        assert len(failures) == 1
        assert failures[0].id == failed.id

        history = await repo.list_for_provider(provider.id)
        assert len(history) == 2
        assert history[0].id == failed.id
        assert healthy.id in {row.id for row in history}


class TestBaseRepositoryOperations:
    @pytest.mark.asyncio
    async def test_get_one_list_exists_count_delete(self, db_session: AsyncSession) -> None:
        repo = ProviderRepository(db_session)
        created = await repo.create(make_provider(name="base-repo-provider"))

        one = await repo.get_one(Provider.name == "base-repo-provider")
        assert one is not None

        rows = await repo.list(Provider.is_active.is_(True), limit=10)
        assert any(row.id == created.id for row in rows)

        assert await repo.exists(Provider.name == "base-repo-provider")
        assert await repo.count(Provider.name == "base-repo-provider") == 1

        await repo.delete(created)
        assert await repo.get_by_id(created.id) is None
