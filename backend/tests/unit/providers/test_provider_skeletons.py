"""
Unit tests for cloud provider registration and HTTP lifecycle.

Covers behavior that is common to every *cloud* provider adapter
(OpenAI/Anthropic/Gemini) rather than to any single vendor's API shape:
that each class self-registers correctly, that the factory can build one
from raw constructor kwargs, and that the shared `httpx.AsyncClient`
lifecycle (owned vs. externally-supplied client) is honored consistently
across all three. Vendor-specific wire-format tests live in the
`test_<provider>_provider.py` files instead.
"""

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

# Each tuple is (provider name, provider class, constructor kwargs). Shared
# across every test below so the same (name, class, kwargs) triple drives
# registration, factory-creation, and HTTP-lifecycle assertions for all
# three cloud providers without duplicating the setup per vendor.
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
    """Return a fresh, empty `ProviderRegistry` isolated from any process-wide default."""
    return ProviderRegistry()


@pytest.mark.parametrize(("name", "provider_cls", "kwargs"), CLOUD_PROVIDERS)
def test_cloud_provider_registers_on_import(
    name: str,
    provider_cls: type,
    kwargs: dict[str, Any],
) -> None:
    """
    Each cloud provider class can be registered under its own name and looked back up.

    Runs once per provider in `CLOUD_PROVIDERS`. `kwargs` is unused here
    (`del kwargs`) since registration only needs the class, not an instance;
    it stays in the parametrize tuple purely so it can be reused by the
    other tests below that *do* need constructor arguments.
    """
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
    """
    The factory instantiates each cloud provider using its vendor-specific constructor kwargs.

    Runs once per provider in `CLOUD_PROVIDERS`, confirming that
    `ProviderFactory.create()` correctly forwards each provider's real
    constructor arguments (e.g. `api_key`, `base_url`) rather than just
    working for the no-argument case.
    """
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
    """
    Using a provider as an async context manager closes its self-created HTTP client on exit.

    Runs once per provider in `CLOUD_PROVIDERS`. When no `http_client` is
    passed in, the provider owns (created) its own `httpx.AsyncClient` and
    must close it itself on `__aexit__` to avoid leaking open connections;
    externally-supplied clients must NOT be closed this way (see the
    companion test below).
    """
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
    """
    `close()` must NOT close an `httpx.AsyncClient` that the caller supplied externally.

    Runs once per provider in `CLOUD_PROVIDERS`. If a caller passes in their
    own shared client (e.g. for connection pooling across multiple provider
    instances), the provider must not take ownership of its lifecycle --
    closing it here would break every other consumer of that shared client.
    """
    del name
    external_client = httpx.AsyncClient(base_url=kwargs["base_url"])
    external_client.aclose = AsyncMock()
    provider = provider_cls(http_client=external_client, **kwargs)

    await provider.close()

    external_client.aclose.assert_not_awaited()
    await external_client.aclose()


def test_openai_accepts_optional_organization() -> None:
    """
    `OpenAIProvider` stores an optional `organization` id for OpenAI's org-scoped billing/API keys.

    This is an OpenAI-specific constructor kwarg (not shared with
    Anthropic/Gemini), so it's tested individually rather than via the
    shared `CLOUD_PROVIDERS` parametrization above.
    """
    provider = OpenAIProvider(api_key="test-key", organization="org-123")

    assert provider._organization == "org-123"  # noqa: SLF001


def test_cloud_provider_rejects_empty_api_key() -> None:
    """
    Constructing a provider with an empty `api_key` fails fast with a clear `ValueError`.

    Catching this at construction time (rather than deferring to the first
    HTTP call) surfaces misconfiguration immediately instead of as a
    confusing 401 from the vendor later.
    """
    with pytest.raises(ValueError, match="api_key"):
        OpenAIProvider(api_key="")
