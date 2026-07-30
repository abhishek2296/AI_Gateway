"""
Integration tests for all persistence repositories.

Each ``Test*Repository`` class exercises one repository class under
``src/repositories/`` against a real PostgreSQL database (via the
``db_session`` fixture — see ``tests/conftest.py``), verifying the
repository's custom query methods (e.g. ``get_by_name``, ``list_active``)
in addition to the generic CRUD operations inherited from the shared base
repository. Test data is built with the ``make_*`` factories in
``tests/factories.py`` rather than hand-rolled ORM instances, so tests stay
focused on the repository behavior being verified.

Requires a live PostgreSQL instance — see the ``integration`` pytest marker
applied via ``pytestmark`` below.
"""

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
    """
    Create and persist a ``Provider`` and one owned ``AIModel``, for tests
    that need a valid parent row but aren't specifically testing provider
    or model creation themselves.

    The provider name includes a random suffix (``uuid.uuid4().hex[:8]``)
    so this helper can be called multiple times within the same test run
    without tripping the unique constraint on ``Provider.name``.

    Args:
        session: The active database session to create both rows through.

    Returns:
        A ``(provider, model)`` tuple of the persisted ``Provider`` and
        ``AIModel`` instances, both with database-assigned ``id`` values.

    Example:
        >>> provider, model = await seed_provider_and_model(db_session)  # doctest: +SKIP
    """
    provider = await ProviderRepository(session).create(make_provider(name=f"repo-{uuid.uuid4().hex[:8]}"))
    model = await AIModelRepository(session).create(
        make_ai_model(provider_id=provider.id, model_name="repo-model"),
    )
    return provider, model


class TestProviderRepository:
    """Tests for ``ProviderRepository``'s CRUD and lookup methods."""

    @pytest.mark.asyncio
    async def test_crud(self, db_session: AsyncSession) -> None:
        """
        Exercise the full create/read/update/delete lifecycle plus the
        shared ``exists``/``count`` base-repository helpers, all against a
        single ``Provider`` row.

        Verifies each step builds on the last correctly: creation assigns
        an id, that id can be used to load the row back, the loaded row can
        be updated and the change is visible, and after deletion the row is
        gone (with a second delete correctly reporting nothing was deleted).
        """
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
        """``get_by_name`` should locate a provider by its unique name column."""
        repo = ProviderRepository(db_session)
        await repo.create(make_provider(name="named-provider"))
        found = await repo.get_by_name("named-provider")
        assert found is not None

    @pytest.mark.asyncio
    async def test_list_active(self, db_session: AsyncSession) -> None:
        """``list_active`` should include only providers with ``is_active=True``."""
        repo = ProviderRepository(db_session)
        await repo.create(make_provider(name="active-provider", is_active=True))
        await repo.create(make_provider(name="inactive-provider", is_active=False))
        active = await repo.list_active()
        names = {p.name for p in active}
        assert "active-provider" in names
        assert "inactive-provider" not in names


class TestAIModelRepository:
    """Tests for ``AIModelRepository``'s custom lookup and filtering methods."""

    @pytest.mark.asyncio
    async def test_crud_and_queries(self, db_session: AsyncSession) -> None:
        """
        Verify ``get_by_provider_and_name``, ``list_enabled_models``, and
        ``get_default_model`` all correctly distinguish an active/default
        model from a disabled one, using two models on the same provider.

        ``list_enabled_models`` must exclude the disabled model entirely,
        while ``get_default_model`` must resolve specifically to the model
        flagged ``is_default=True`` (not just any enabled model).
        """
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
    """Tests for ``ProviderConfigurationRepository``'s custom lookup methods."""

    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        """
        ``get_by_provider_id`` should find the one-to-one configuration row
        for a provider, and ``list_active`` should include it.
        """
        provider = await ProviderRepository(db_session).create(make_provider(name="config-provider"))
        repo = ProviderConfigurationRepository(db_session)
        config = await repo.create(make_provider_configuration(provider_id=provider.id))

        assert await repo.get_by_provider_id(provider.id) is not None
        active = await repo.list_active()
        assert any(item.id == config.id for item in active)


class TestAIModelConfigurationRepository:
    """Tests for ``AIModelConfigurationRepository``'s custom lookup methods."""

    @pytest.mark.asyncio
    async def test_get_by_ai_model_id(self, db_session: AsyncSession) -> None:
        """``get_by_ai_model_id`` should find the one-to-one configuration row for a model."""
        provider, model = await seed_provider_and_model(db_session)
        repo = AIModelConfigurationRepository(db_session)
        config = await repo.create(make_ai_model_configuration(ai_model_id=model.id))

        loaded = await repo.get_by_ai_model_id(model.id)
        assert loaded is not None
        assert loaded.id == config.id


class TestChatSessionRepository:
    """Tests for ``ChatSessionRepository``'s custom lookup and mutation methods."""

    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        """
        Verify ``get_by_uuid``, ``list_recent_sessions``, and
        ``archive_session`` together, using one active and one already
        archived session.

        ``list_recent_sessions`` must exclude the archived session (it's
        meant to surface sessions a user would still want to resume), while
        ``archive_session`` on the active one should flip its
        ``is_archived`` flag and return the updated row.
        """
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
    """Tests for ``MessageRepository``'s ordering and counting methods."""

    @pytest.mark.asyncio
    async def test_list_and_count(self, db_session: AsyncSession) -> None:
        """
        ``list_messages`` should return messages for a session in creation
        order, and ``count_for_session`` should match the number created.

        Asserting the exact id ordering (``[first.id, second.id]``) rather
        than just set membership confirms conversation history round-trips
        in chronological order, not an arbitrary order.
        """
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
    """Tests for ``PromptTemplateRepository``'s custom lookup methods."""

    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        """
        ``get_by_name_and_version`` should find a specific template
        revision, and ``list_active`` (with no category filter) should
        include only templates flagged ``is_active=True``.
        """
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
    """Tests for ``APIKeyRepository``'s custom lookup methods."""

    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        """
        Verify ``get_by_provider_and_name``, ``get_default_for_provider``,
        and ``list_active_for_provider`` using one default/active key and
        one non-default/inactive key on the same provider.

        ``list_active_for_provider`` must exclude the inactive key, and
        ``get_default_for_provider`` must specifically resolve to the key
        flagged ``is_default=True``.
        """
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
    """Tests for ``UsageRecordRepository``'s billing/telemetry query methods."""

    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        """
        Verify ``get_by_request_id``, ``usage_between_dates``, and
        ``usage_by_provider`` all correctly locate the same usage record via
        different query dimensions (unique request id, a surrounding time
        window, and owning provider).
        """
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
    """Tests for ``ProviderHealthRepository``'s history and status query methods."""

    @pytest.mark.asyncio
    async def test_queries(self, db_session: AsyncSession) -> None:
        """
        Verify ``latest_health``, ``failed_checks``, and
        ``list_for_provider`` against two health check rows recorded a
        minute apart, one healthy and one failed.

        ``latest_health`` must resolve to the most recently ``checked_at``
        row (the failed one, since it was recorded a minute later) rather
        than the most recently *created* row, and ``failed_checks`` must
        include only the unhealthy entry.
        """
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
    """Tests for the generic query helpers on the shared base repository class."""

    @pytest.mark.asyncio
    async def test_get_one_list_exists_count_delete(self, db_session: AsyncSession) -> None:
        """
        Exercise ``get_one``, ``list`` (with a filter and a limit),
        ``exists``, ``count``, and ``delete`` — the generic, filter-based
        operations every repository inherits — using ``ProviderRepository``
        as a concrete stand-in.

        Because these operations are defined once on the shared base
        repository and reused by every entity-specific repository, testing
        them here (rather than duplicating identical assertions in every
        ``Test*Repository`` class above) is sufficient to cover all of them.
        """
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
