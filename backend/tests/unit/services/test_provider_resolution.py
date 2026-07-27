"""Unit tests for provider resolution components."""

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
    return AsyncMock()


@pytest.fixture
def model_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def api_key_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def config_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def resolver(
    provider_repo: AsyncMock,
    model_repo: AsyncMock,
    api_key_repo: AsyncMock,
    config_repo: AsyncMock,
) -> ProviderResolver:
    return ProviderResolver(
        provider_repository=provider_repo,
        model_repository=model_repo,
        api_key_repository=api_key_repo,
        configuration_repository=config_repo,
        settings=_settings(),
    )


@pytest.mark.asyncio
async def test_resolve_provider_from_request_override(resolver: ProviderResolver, provider_repo: AsyncMock) -> None:
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
    provider_repo.get_by_name.return_value = _provider(name="ollama", is_active=False)

    with pytest.raises(ProviderDisabledError):
        await resolver.resolve(request_provider="ollama")


@pytest.mark.asyncio
async def test_unable_to_resolve_when_all_strategies_empty(resolver: ProviderResolver) -> None:
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
