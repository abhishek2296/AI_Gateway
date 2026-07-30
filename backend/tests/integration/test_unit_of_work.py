"""
Unit of Work transaction and session coordination tests.

``AsyncUnitOfWork`` (see ``src/unit_of_work.py``) bundles multiple
repositories behind a single shared session so a caller can perform several
related writes (e.g. create a provider *and* its default model) and commit
or roll them all back together, atomically. These tests verify:

- Repositories exposed by the same Unit of Work share one session.
- Explicit ``commit()`` durably persists changes, visible from another,
  independent session.
- ``rollback()`` (explicit or via context-manager exception handling)
  discards uncommitted changes.
- The Unit of Work does *not* auto-commit on normal context-manager exit —
  callers must call ``commit()`` explicitly.
- ``close()`` correctly distinguishes an externally managed session (never
  closed by the Unit of Work) from one the Unit of Work owns.

Requires a live PostgreSQL instance — see the ``integration`` pytest marker
applied via ``pytestmark`` below, and the ``uow``/``db_session``/
``committed_session`` fixtures in ``tests/conftest.py``.
"""

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
    """
    Every repository exposed by a given ``AsyncUnitOfWork`` instance must be
    bound to the exact same ``AsyncSession`` object.

    This is the core guarantee that makes the Unit of Work pattern useful:
    without a shared session, writes made through ``uow.providers`` and
    ``uow.ai_models`` couldn't be committed or rolled back together as one
    atomic operation.
    """
    assert uow.providers.session is uow.ai_models.session
    assert uow.providers.session is uow.messages.session


@pytest.mark.asyncio
async def test_explicit_commit_persists(
    test_engine: AsyncEngine,
    committed_session: AsyncSession,
) -> None:
    """
    Calling ``uow.commit()`` must durably persist writes so they are visible
    from a completely independent session/connection afterward.

    Uses ``committed_session`` (not the rollback-based ``db_session``)
    because this test needs the write to actually reach the database, then
    opens a brand-new ``verify_session`` from the shared engine to prove the
    data is visible outside the Unit of Work's own session — a true
    cross-session durability check, not just "the same session can see its
    own uncommitted writes."
    """
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
    """
    Calling ``uow.rollback()`` explicitly must discard writes made through
    that Unit of Work, so a subsequent read on the same session no longer
    sees them.
    """
    uow = AsyncUnitOfWork(db_session, close_session=False)
    await uow.providers.create(make_provider(name="uow-rollback-provider"))
    await uow.rollback()

    assert await uow.providers.get_by_name("uow-rollback-provider") is None


@pytest.mark.asyncio
async def test_context_manager_rolls_back_on_exception(db_session: AsyncSession) -> None:
    """
    If an exception propagates out of an ``async with AsyncUnitOfWork(...)``
    block, the Unit of Work must roll back any writes made inside it before
    the exception continues propagating.

    This is the safety net the context-manager protocol provides: callers
    don't need a manual ``try/except/rollback`` around every Unit of Work
    usage — raising *any* exception inside the block is enough to guarantee
    partial work is discarded.
    """
    with pytest.raises(RuntimeError):
        async with AsyncUnitOfWork(db_session, close_session=False) as uow:
            await uow.providers.create(make_provider(name="uow-exception-provider"))
            raise RuntimeError("trigger rollback")

    assert await ProviderRepository(db_session).get_by_name("uow-exception-provider") is None


@pytest.mark.asyncio
async def test_context_manager_does_not_auto_commit(db_session: AsyncSession) -> None:
    """
    Exiting the ``async with AsyncUnitOfWork(...)`` block *without* an
    exception must not implicitly commit — callers are required to call
    ``uow.commit()`` themselves if they want the changes to persist.

    This is intentional API design (not a bug): auto-committing on clean
    exit would make it impossible to compose multiple Unit of Work blocks
    into one larger caller-controlled transaction. The test confirms rows
    created inside the block remain visible only via flush (same
    transaction) and disappear once the outer ``db_session`` is explicitly
    rolled back.
    """
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
    """
    Writes made through two different repositories on the same Unit of Work
    (``uow.providers`` and ``uow.ai_models``) must commit together as a
    single atomic transaction and both be durably visible afterward.

    This is the primary real-world use case for the Unit of Work pattern:
    creating a provider and its default model as one logical operation.
    Verified via an independent ``verify_session`` (as in
    ``test_explicit_commit_persists``) plus a cross-check that the loaded
    model's ``provider_id`` correctly points back at the loaded provider.
    """
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
    """
    When constructed with ``close_session=False`` (an externally managed
    session, e.g. one owned by the ``db_session`` fixture), ``uow.close()``
    must NOT actually close the underlying session.

    Proven indirectly: after calling ``close()``, the session is used again
    to create a provider, which would fail if the session had really been
    closed. This matters because ``db_session``'s own teardown still needs
    to roll back and close the session itself afterward — a Unit of Work
    that isn't the session's owner must never interfere with that.
    """
    uow = AsyncUnitOfWork(db_session, close_session=False)
    await uow.close()
    provider = await uow.providers.create(make_provider(name="after-close-check"))
    assert provider.id is not None


@pytest.mark.asyncio
async def test_close_closes_owned_session(test_engine: AsyncEngine) -> None:
    """
    When constructed with ``close_session=True`` (the Unit of Work owns its
    session, created via its own ``session_factory()`` call rather than
    injected from a fixture), ``uow.close()`` must actually close it and
    flip the internal ``_closed`` flag.

    This is the complementary case to
    ``test_close_skips_when_session_externally_managed``, confirming
    ``close_session`` correctly toggles ownership-based cleanup behavior.
    """
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    session = session_factory()
    uow = AsyncUnitOfWork(session, close_session=True)
    await uow.close()
    assert uow._closed is True
