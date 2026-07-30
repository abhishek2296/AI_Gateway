"""
Unit tests for provider resolution components.

Covers three collaborating pieces of the resolution pipeline (see
``src/services/provider_resolver.py``, ``src/services/provider_config_resolver.py``,
and ``src/core/provider_resolution_cache.py``):

- ``ProviderResolver`` — decides which provider/model name to use for a
  request by walking an ordered chain of strategies (explicit request
  override -> database default -> settings fallback -> hardcoded fallback),
  and resolves the applicable API key reference for a provider.
- ``ProviderConfigResolver`` — turns a resolved provider/configuration/API
  key selection into the keyword arguments a provider client factory needs
  (base URL, timeout, API key value, etc.).
- ``ProviderResolutionCache`` — a small TTL cache in front of resolution so
  repeated requests with the same explicit provider/model don't re-run the
  full strategy chain every time.

All repository dependencies are mocked with ``AsyncMock`` so these tests
exercise only the resolution *logic*, never a real database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import Settings
from src.core.provider_resolution_cache import ProviderResolutionCache
from src.models.ai_model import AIModel
from src.models.api_key import APIKey
from src.models.provider import Provider
from src.models.provider_configuration import ProviderConfiguration
from src.services.provider_config_resolver import ProviderConfigResolver
from src.services.provider_resolver import (
    ProviderDisabledError,
    ProviderResolutionError,
    ProviderResolver,
)
from src.services.resolution_types import ResolutionSource, ResolvedConfiguration


def _settings(**overrides: object) -> Settings:
    """
    Build a ``Settings`` instance with sane test defaults, overriding as needed.

    Centralizing the boilerplate fields (DB URL, default provider/model,
    Ollama host, timeout) here means each test only has to specify the one
    or two settings it actually cares about (e.g. ``DEFAULT_PROVIDER=""`` to
    exercise the hardcoded-fallback path).

    Args:
        **overrides: Field name/value pairs that override the defaults,
            e.g. ``DEFAULT_PROVIDER="custom"``.

    Returns:
        A fully constructed ``Settings`` object suitable for injecting into
        a ``ProviderResolver`` under test.

    Example:
        >>> settings = _settings(DEFAULT_PROVIDER="")
        >>> settings.DEFAULT_PROVIDER
        ''
    """
    defaults = {
        "DATABASE_URL": "postgresql+asyncpg://ai_app:pass@localhost:5432/db",
        "DEFAULT_PROVIDER": "settings-provider",
        "DEFAULT_MODEL": "settings-model",
        "OLLAMA_HOST": "http://localhost:11434",
        "TIMEOUT": 60,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _provider(
    *,
    name: str = "ollama",
    is_active: bool = True,
    base_url: str | None = "http://db-ollama:11434",
    provider_id: int = 1,
) -> Provider:
    """
    Build an in-memory ``Provider`` ORM instance for resolver tests.

    This does not touch the database — it just constructs a plain ``Provider``
    object with the fields the resolution logic reads (``name``,
    ``is_active``, ``base_url``), letting a test control exactly which
    resolution branch is exercised (e.g. ``is_active=False`` to trigger
    ``ProviderDisabledError``).

    Args:
        name: Provider name/identifier, e.g. ``"ollama"``. Also used as
            ``display_name`` and ``provider_type`` for simplicity.
        is_active: Whether the provider should appear enabled. Set to
            ``False`` to test the disabled-provider error path.
        base_url: Provider base URL stored on the row itself (as opposed to
            an override in ``ProviderConfiguration``). ``None`` simulates a
            provider with no configured base URL.
        provider_id: Primary key to assign, referenced by related fixtures
            (e.g. ``AIModel.provider_id``).

    Returns:
        A ``Provider`` instance (not persisted) with the given attributes.

    Example:
        >>> provider = _provider(name="openai", is_active=True)
        >>> provider.name
        'openai'
    """
    return Provider(
        id=provider_id,
        name=name,
        display_name=name,
        provider_type=name,
        base_url=base_url,
        is_active=is_active,
    )


def _model(
    *,
    provider_id: int = 1,
    model_name: str = "db-default-model",
    is_default: bool = True,
) -> AIModel:
    """
    Build an in-memory ``AIModel`` ORM instance for resolver tests.

    Args:
        provider_id: Foreign key to the owning ``Provider``. Must match the
            ``provider_id`` used when building the corresponding ``_provider``.
        model_name: The model identifier the database would report as the
            provider's default model.
        is_default: Whether this model should be flagged as the provider's
            default. Kept ``True`` by default since these tests primarily
            exercise the "resolve the default model" code path.

    Returns:
        An ``AIModel`` instance (not persisted) with ``is_active=True``.

    Example:
        >>> model = _model(provider_id=1, model_name="qwen3:8b")
        >>> model.is_default
        True
    """
    return AIModel(
        id=1,
        provider_id=provider_id,
        model_name=model_name,
        display_name=model_name,
        is_default=is_default,
        is_active=True,
    )


@pytest.fixture
def provider_repo() -> AsyncMock:
    """Function-scoped mock standing in for ``ProviderRepository``.

    Returns:
        A fresh ``AsyncMock`` per test so return values configured by one
        test (e.g. ``get_by_name.return_value = ...``) never leak into
        another test.
    """
    return AsyncMock()


@pytest.fixture
def model_repo() -> AsyncMock:
    """Function-scoped mock standing in for ``AIModelRepository``.

    Returns:
        A fresh ``AsyncMock`` used to stub methods like
        ``get_default_model`` without hitting a real database.
    """
    return AsyncMock()


@pytest.fixture
def api_key_repo() -> AsyncMock:
    """Function-scoped mock standing in for ``APIKeyRepository``.

    Returns:
        A fresh ``AsyncMock`` used to stub methods like
        ``get_default_for_provider`` without hitting a real database.
    """
    return AsyncMock()


@pytest.fixture
def config_repo() -> AsyncMock:
    """Function-scoped mock standing in for ``ProviderConfigurationRepository``.

    Returns:
        A fresh ``AsyncMock`` used to stub methods like
        ``get_by_provider_id`` without hitting a real database.
    """
    return AsyncMock()


@pytest.fixture
def resolver(
    provider_repo: AsyncMock,
    model_repo: AsyncMock,
    api_key_repo: AsyncMock,
    config_repo: AsyncMock,
) -> ProviderResolver:
    """
    Build a ``ProviderResolver`` wired to the mocked repositories above.

    Using the mock fixtures (rather than constructing new ``AsyncMock()``
    instances inline) lets each test both configure a repository's behavior
    *and* assert against it via the same fixture object.

    Args:
        provider_repo: Mocked provider repository, injected by pytest.
        model_repo: Mocked model repository, injected by pytest.
        api_key_repo: Mocked API key repository, injected by pytest.
        config_repo: Mocked provider configuration repository, injected by
            pytest.

    Returns:
        A ``ProviderResolver`` configured with default test ``Settings``
        (see ``_settings()``), ready to have its repository mocks
        programmed per test.
    """
    return ProviderResolver(
        provider_repository=provider_repo,
        model_repository=model_repo,
        api_key_repository=api_key_repo,
        configuration_repository=config_repo,
        settings=_settings(),
    )


@pytest.mark.asyncio
async def test_resolve_provider_from_request_override(resolver: ProviderResolver, provider_repo: AsyncMock) -> None:
    """
    An explicit ``request_provider``/``request_model`` should win over any
    database or settings default.

    This verifies the resolution chain's highest-priority strategy: if the
    caller of the API explicitly names a provider/model in the request, that
    choice must be honored as-is (source ``REQUEST``) rather than falling
    through to the database default, even though a database default model
    is also configured here (``db-model``).
    """
    provider_repo.get_by_name.return_value = _provider(name="ollama")
    model_repo = resolver._model_repository
    model_repo.get_default_model.return_value = _model(model_name="db-model")
    config_repo = resolver._configuration_repository
    config_repo.get_by_provider_id.return_value = None
    api_key_repo = resolver._api_key_repository
    api_key_repo.get_default_for_provider.return_value = None

    result = await resolver.resolve(request_provider="ollama", request_model="request-model")

    assert result.provider_name == "ollama"
    assert result.model_name == "request-model"
    assert result.provider_source == ResolutionSource.REQUEST
    assert result.model_source == ResolutionSource.REQUEST


@pytest.mark.asyncio
async def test_resolve_provider_from_database_default(
    resolver: ProviderResolver,
    provider_repo: AsyncMock,
) -> None:
    """
    With no request override, the resolver should use the provider/model
    flagged as default in the database.

    This is the second strategy in the chain: absent an explicit request
    value, the resolver should query for a database-configured default
    provider (``get_default_provider``) and default model
    (``get_default_model``), and report the source as ``DATABASE`` so
    callers/observability can distinguish this from a settings-based
    fallback.
    """
    provider_repo.get_default_provider.return_value = _provider(name="db-provider")
    provider_repo.get_by_name.return_value = _provider(name="db-provider")
    model_repo = resolver._model_repository
    model_repo.get_default_model.return_value = _model(model_name="db-default-model")
    config_repo = resolver._configuration_repository
    config_repo.get_by_provider_id.return_value = None
    api_key_repo = resolver._api_key_repository
    api_key_repo.get_default_for_provider.return_value = None

    result = await resolver.resolve()

    assert result.provider_name == "db-provider"
    assert result.model_name == "db-default-model"
    assert result.provider_source == ResolutionSource.DATABASE
    assert result.model_source == ResolutionSource.DATABASE


@pytest.mark.asyncio
async def test_resolve_provider_settings_fallback(
    resolver: ProviderResolver,
    provider_repo: AsyncMock,
) -> None:
    """
    When neither a request override nor a database default exists, fall
    back to the values configured in application ``Settings``.

    Both repository lookups return ``None`` here (no database default
    provider, and looking the settings-provider name up by name also
    fails), so the resolver must fall back to ``DEFAULT_PROVIDER``/
    ``DEFAULT_MODEL`` from settings and label the source as ``SETTINGS``.
    """
    provider_repo.get_default_provider.return_value = None
    provider_repo.get_by_name.return_value = None

    result = await resolver.resolve()

    assert result.provider_name == "settings-provider"
    assert result.model_name == "settings-model"
    assert result.provider_source == ResolutionSource.SETTINGS
    assert result.model_source == ResolutionSource.SETTINGS


@pytest.mark.asyncio
async def test_resolve_provider_hardcoded_fallback(
    provider_repo: AsyncMock,
    model_repo: AsyncMock,
    api_key_repo: AsyncMock,
    config_repo: AsyncMock,
) -> None:
    """
    If ``Settings.DEFAULT_PROVIDER`` is itself blank, the resolver must fall
    back to a hardcoded provider name (``"ollama"``) rather than resolving
    to an empty string or raising.

    This is the last-resort strategy in the chain: it guarantees
    ``resolve()`` always returns *some* usable provider name, even in a
    misconfigured environment where nothing else is available.
    """
    settings = _settings(DEFAULT_PROVIDER="")
    resolver = ProviderResolver(
        provider_repository=provider_repo,
        model_repository=model_repo,
        api_key_repository=api_key_repo,
        configuration_repository=config_repo,
        settings=settings,
    )
    provider_repo.get_default_provider.return_value = None
    provider_repo.get_by_name.return_value = None

    result = await resolver.resolve()

    assert result.provider_name == "ollama"
    assert result.provider_source == ResolutionSource.FALLBACK


@pytest.mark.asyncio
async def test_resolve_model_settings_when_no_database_default(
    resolver: ProviderResolver,
    provider_repo: AsyncMock,
) -> None:
    """
    Model resolution falls back to ``Settings.DEFAULT_MODEL`` when the
    chosen provider has no default model configured in the database.

    Note the provider itself *is* resolved successfully here (via request
    override), but ``model_repo.get_default_model`` returns ``None`` —
    this isolates model-source fallback behavior from provider-source
    fallback behavior, confirming each resolves independently.
    """
    provider_repo.get_by_name.return_value = _provider(name="ollama")
    model_repo = resolver._model_repository
    model_repo.get_default_model.return_value = None
    config_repo = resolver._configuration_repository
    config_repo.get_by_provider_id.return_value = None
    api_key_repo = resolver._api_key_repository
    api_key_repo.get_default_for_provider.return_value = None

    result = await resolver.resolve(request_provider="ollama")

    assert result.model_name == "settings-model"
    assert result.model_source == ResolutionSource.SETTINGS


@pytest.mark.asyncio
async def test_disabled_provider_raises(
    resolver: ProviderResolver,
    provider_repo: AsyncMock,
) -> None:
    """
    Resolving to a provider that exists but has ``is_active=False`` must
    raise ``ProviderDisabledError`` rather than silently routing to a
    disabled backend.

    This is a safety check: an operator disabling a provider (e.g. during
    an outage) must reliably block traffic to it even if a client
    explicitly asks for it by name.
    """
    provider_repo.get_by_name.return_value = _provider(name="ollama", is_active=False)

    with pytest.raises(ProviderDisabledError):
        await resolver.resolve(request_provider="ollama")


@pytest.mark.asyncio
async def test_unable_to_resolve_when_all_strategies_empty(resolver: ProviderResolver) -> None:
    """
    If every configured provider-name strategy returns ``None`` (simulating
    a pathological configuration with no fallback), ``resolve()`` must raise
    ``ProviderResolutionError`` instead of returning an invalid/empty result.

    The test directly overwrites the resolver's internal
    ``_provider_name_strategies`` tuple with three no-op strategies (none of
    which include the real hardcoded fallback) to force this otherwise
    hard-to-reach edge case.
    """

    async def empty() -> str | None:
        return None

    resolver._provider_name_strategies = (
        (ResolutionSource.SETTINGS, empty),
        (ResolutionSource.SETTINGS, empty),
        (ResolutionSource.FALLBACK, empty),
    )

    with pytest.raises(ProviderResolutionError):
        await resolver.resolve()


def test_config_resolver_builds_ollama_kwargs() -> None:
    """
    ``ProviderConfigResolver.build`` should prefer the endpoint/timeout from
    an explicit ``ProviderConfiguration`` row over the provider's own
    ``base_url`` column.

    The provider row has ``base_url="http://db-host:11434"`` but the
    configuration row has ``endpoint="http://config-endpoint:11434"`` and
    ``timeout_seconds=45`` — the resolved kwargs must use the configuration
    values, confirming configuration-level overrides take precedence.
    """
    provider = _provider(base_url="http://db-host:11434")
    configuration = ProviderConfiguration(
        id=1,
        provider_id=1,
        endpoint="http://config-endpoint:11434",
        timeout_seconds=45,
        verify_ssl=True,
    )
    selection = MagicMock(
        provider_name="ollama",
        provider=provider,
        configuration=configuration,
        api_key=None,
    )
    resolver = ProviderConfigResolver(_settings())

    kwargs = resolver.build(selection)

    assert kwargs["base_url"] == "http://config-endpoint:11434"
    assert kwargs["timeout"] == 45.0


def test_config_resolver_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When a resolved ``APIKey`` reference names an environment variable
    (``api_key_env``), the config resolver must read the actual secret value
    from the process environment rather than storing/returning it from the
    database.

    This confirms the "reference, not value" pattern for API key storage:
    the database only ever holds *which* env var to read, never the secret
    itself, and ``monkeypatch.setenv`` here stands in for that env var being
    set in the real deployment environment.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    provider = _provider(name="openai")
    api_key = APIKey(
        id=1,
        provider_id=1,
        name="primary",
        key_identifier="prod",
        api_key_env="OPENAI_API_KEY",
        is_default=True,
        is_active=True,
    )
    selection = MagicMock(
        provider_name="openai",
        provider=provider,
        configuration=None,
        api_key=api_key,
    )
    resolver = ProviderConfigResolver(_settings())

    kwargs = resolver.build(selection)

    assert kwargs["api_key"] == "secret-value"


def test_resolution_cache_hit_and_miss() -> None:
    """
    A cache miss (``get`` before any ``set``) returns ``None``; after
    ``set``, ``get`` with the same key returns the cached value.

    ``None, None`` (no explicit request provider/model) is used as the
    cache key here to confirm the cache correctly treats "resolve using
    defaults" as its own distinct, cacheable key rather than special-casing
    it.
    """
    cache = ProviderResolutionCache(ttl_seconds=60.0)
    config = ResolvedConfiguration(
        provider_name="ollama",
        model_name="qwen3:8b",
        factory_kwargs={"base_url": "http://localhost:11434"},
        provider_source=ResolutionSource.SETTINGS,
        model_source=ResolutionSource.SETTINGS,
    )

    assert cache.get(None, None) is None
    cache.set(None, None, config)
    assert cache.get(None, None) == config


def test_resolution_cache_expiration(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A cached entry must no longer be returned once its TTL has elapsed.

    Rather than sleeping in the test (slow and flaky), this patches
    ``time.monotonic`` inside the cache module with a controllable fake
    clock: the entry is cached at simulated time ``0.0`` with a 1-second
    TTL, then the clock is advanced past the TTL to ``2.0`` before checking
    the cache expired as expected.
    """
    current = {"value": 0.0}

    def fake_monotonic() -> float:
        return current["value"]

    monkeypatch.setattr("src.core.provider_resolution_cache.time.monotonic", fake_monotonic)
    cache = ProviderResolutionCache(ttl_seconds=1.0)
    config = ResolvedConfiguration(
        provider_name="ollama",
        model_name="qwen3:8b",
        factory_kwargs={},
        provider_source=ResolutionSource.SETTINGS,
        model_source=ResolutionSource.SETTINGS,
    )
    cache.set(None, None, config)
    current["value"] = 2.0
    assert cache.get(None, None) is None


@pytest.mark.asyncio
async def test_resolve_api_key_returns_reference(resolver: ProviderResolver, api_key_repo: AsyncMock) -> None:
    """
    ``resolve_api_key`` should surface the resolved key's env var name and
    identifier without ever exposing the actual secret value.

    Confirms the returned object carries just enough metadata
    (``env_var``, ``key_identifier``) for a caller to look up the real
    secret later, matching the "reference, not value" storage pattern
    verified more directly by ``test_config_resolver_reads_api_key_from_env``.
    """
    provider = _provider()
    api_key_repo.get_default_for_provider.return_value = APIKey(
        id=1,
        provider_id=1,
        name="primary",
        key_identifier="prod",
        api_key_env="OLLAMA_API_KEY",
        is_default=True,
        is_active=True,
        expires_at=datetime.now(tz=UTC),
    )

    resolved = await resolver.resolve_api_key(provider)

    assert resolved is not None
    assert resolved.env_var == "OLLAMA_API_KEY"
    assert resolved.key_identifier == "prod"
