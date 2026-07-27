"""Unit tests for cloud provider registration and HTTP lifecycle."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry
from src.providers.anthropic import AnthropicProvider
from src.providers.gemini import GeminiProvider
from src.providers.openai import OpenAIProvider

CLOUD_PROVIDERS: list[tuple[str, type, dict[str, Any]]] = [
    (
        "openai",
        OpenAIProvider,
        {"api_key": "test-key", "base_url": "https://api.openai.com/v1"},
    ),
    (
        "anthropic",
        AnthropicProvider,
        {"api_key": "test-key", "base_url": "https://api.anthropic.com"},
    ),
    (
        "gemini",
        GeminiProvider,
        {"api_key": "test-key", "base_url": "https://generativelanguage.googleapis.com"},
    ),
]


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


@pytest.mark.parametrize(("name", "provider_cls", "kwargs"), CLOUD_PROVIDERS)
def test_cloud_provider_registers_on_import(
    name: str,
    provider_cls: type,
    kwargs: dict[str, Any],
) -> None:
    del kwargs
    isolated = ProviderRegistry()
    isolated.register(provider_cls)
    assert isolated.exists(name)
    assert isolated.get(name) is provider_cls


@pytest.mark.parametrize(("name", "provider_cls", "kwargs"), CLOUD_PROVIDERS)
def test_factory_creates_cloud_provider(
    registry: ProviderRegistry,
    name: str,
    provider_cls: type,
    kwargs: dict[str, Any],
) -> None:
    registry.register(provider_cls)
    factory = ProviderFactory(registry)

    provider = factory.create(name, **kwargs)

    assert isinstance(provider, provider_cls)


@pytest.mark.parametrize(("name", "provider_cls", "kwargs"), CLOUD_PROVIDERS)
@pytest.mark.asyncio
async def test_cloud_provider_context_manager_closes_owned_client(
    name: str,
    provider_cls: type,
    kwargs: dict[str, Any],
) -> None:
    del name
    async with provider_cls(**kwargs) as provider:
        provider._client.aclose = AsyncMock()  # noqa: SLF001

    provider._client.aclose.assert_awaited_once()  # noqa: SLF001


@pytest.mark.parametrize(("name", "provider_cls", "kwargs"), CLOUD_PROVIDERS)
@pytest.mark.asyncio
async def test_cloud_provider_close_does_not_close_external_client(
    name: str,
    provider_cls: type,
    kwargs: dict[str, Any],
) -> None:
    del name
    external_client = httpx.AsyncClient(base_url=kwargs["base_url"])
    external_client.aclose = AsyncMock()
    provider = provider_cls(http_client=external_client, **kwargs)

    await provider.close()

    external_client.aclose.assert_not_awaited()
    await external_client.aclose()


def test_openai_accepts_optional_organization() -> None:
    provider = OpenAIProvider(api_key="test-key", organization="org-123")

    assert provider._organization == "org-123"  # noqa: SLF001


def test_cloud_provider_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAIProvider(api_key="")
