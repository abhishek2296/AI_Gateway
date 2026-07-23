"""Unit of Work transaction and session coordination tests."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.repositories.provider_repository import ProviderRepository
from src.unit_of_work import AsyncUnitOfWork
from tests.conftest import truncate_application_tables
from tests.factories import make_ai_model, make_provider

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_repositories_share_session(uow: AsyncUnitOfWork) -> None:
    assert uow.providers.session is uow.ai_models.session
    assert uow.providers.session is uow.messages.session


@pytest.mark.asyncio
async def test_explicit_commit_persists(
    test_engine: AsyncEngine,
    committed_session: AsyncSession,
) -> None:
    uow = AsyncUnitOfWork(committed_session, close_session=False)
    provider = await uow.providers.create(make_provider(name="uow-commit-provider"))
    await uow.ai_models.create(
        make_ai_model(provider_id=provider.id, model_name="uow-model"),
    )
    await uow.commit()

    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with session_factory() as verify_session:
        loaded = await ProviderRepository(verify_session).get_by_name("uow-commit-provider")
        assert loaded is not None

    await truncate_application_tables(committed_session)


@pytest.mark.asyncio
async def test_rollback_discards_uncommitted_changes(db_session: AsyncSession) -> None:
    uow = AsyncUnitOfWork(db_session, close_session=False)
    await uow.providers.create(make_provider(name="uow-rollback-provider"))
    await uow.rollback()

    assert await uow.providers.get_by_name("uow-rollback-provider") is None


@pytest.mark.asyncio
async def test_context_manager_rolls_back_on_exception(db_session: AsyncSession) -> None:
    with pytest.raises(RuntimeError):
        async with AsyncUnitOfWork(db_session, close_session=False) as uow:
            await uow.providers.create(make_provider(name="uow-exception-provider"))
            raise RuntimeError("trigger rollback")

    assert await ProviderRepository(db_session).get_by_name("uow-exception-provider") is None


@pytest.mark.asyncio
async def test_context_manager_does_not_auto_commit(db_session: AsyncSession) -> None:
    async with AsyncUnitOfWork(db_session, close_session=False) as uow:
        created = await uow.providers.create(make_provider(name="uow-no-auto-commit"))
        created_id = created.id

    # Flushed rows remain visible in the same transaction until rollback.
    assert await ProviderRepository(db_session).get_by_id(created_id) is not None
    await db_session.rollback()
    assert await ProviderRepository(db_session).get_by_name("uow-no-auto-commit") is None


@pytest.mark.asyncio
async def test_multi_repository_transaction(
    test_engine: AsyncEngine,
    committed_session: AsyncSession,
) -> None:
    uow = AsyncUnitOfWork(committed_session, close_session=False)
    provider = await uow.providers.create(make_provider(name="uow-multi-provider"))
    model = await uow.ai_models.create(make_ai_model(provider_id=provider.id, model_name="multi-model"))
    await uow.commit()

    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with session_factory() as verify_session:
        verify_uow = AsyncUnitOfWork(verify_session, close_session=False)
        loaded_provider = await verify_uow.providers.get_by_id(provider.id)
        loaded_model = await verify_uow.ai_models.get_by_id(model.id)
        assert loaded_provider is not None
        assert loaded_model is not None
        assert loaded_model.provider_id == loaded_provider.id

    await truncate_application_tables(committed_session)


@pytest.mark.asyncio
async def test_close_skips_when_session_externally_managed(db_session: AsyncSession) -> None:
    uow = AsyncUnitOfWork(db_session, close_session=False)
    await uow.close()
    provider = await uow.providers.create(make_provider(name="after-close-check"))
    assert provider.id is not None


@pytest.mark.asyncio
async def test_close_closes_owned_session(test_engine: AsyncEngine) -> None:
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    session = session_factory()
    uow = AsyncUnitOfWork(session, close_session=True)
    await uow.close()
    assert uow._closed is True
