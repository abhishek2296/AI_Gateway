"""Unit tests for provider registry and factory."""

from __future__ import annotations

import pytest

from src.providers.exceptions import ProviderError, ProviderNotFoundError
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry, register_provider
from tests.unit.providers.mock_provider import MockProvider


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


@pytest.fixture
def factory(registry: ProviderRegistry) -> ProviderFactory:
    return ProviderFactory(registry)


def test_successful_registration(registry: ProviderRegistry) -> None:
    registry.register(MockProvider)

    assert registry.exists("mock")
    assert registry.get("mock") is MockProvider
    assert registry.available() == ("mock",)


def test_duplicate_registration_raises(registry: ProviderRegistry) -> None:
    registry.register(MockProvider)

    with pytest.raises(ProviderError, match="already registered"):
        registry.register(MockProvider)


def test_unregister(registry: ProviderRegistry) -> None:
    registry.register(MockProvider)
    registry.unregister("mock")

    assert not registry.exists("mock")
    assert registry.available() == ()


def test_unregister_missing_raises(registry: ProviderRegistry) -> None:
    with pytest.raises(ProviderNotFoundError):
        registry.unregister("missing")


def test_create_instantiates_provider(
    registry: ProviderRegistry,
    factory: ProviderFactory,
) -> None:
    registry.register(MockProvider)

    provider = factory.create("mock", endpoint="http://test")

    assert isinstance(provider, MockProvider)
    assert provider.endpoint == "http://test"


def test_create_returns_fresh_instances(registry: ProviderRegistry, factory: ProviderFactory) -> None:
    registry.register(MockProvider)

    first = factory.create("mock")
    second = factory.create("mock")

    assert first is not second


def test_create_missing_provider_raises(factory: ProviderFactory) -> None:
    with pytest.raises(ProviderNotFoundError):
        factory.create("missing")


def test_get_missing_raises(registry: ProviderRegistry) -> None:
    with pytest.raises(ProviderNotFoundError):
        registry.get("missing")


def test_available_sorted(registry: ProviderRegistry) -> None:
    class AlphaProvider(MockProvider):
        provider_name = "alpha"

    class ZetaProvider(MockProvider):
        provider_name = "zeta"

    registry.register(ZetaProvider)
    registry.register(AlphaProvider)

    assert registry.available() == ("alpha", "zeta")


def test_clear(registry: ProviderRegistry) -> None:
    registry.register(MockProvider)
    registry.clear()

    assert registry.available() == ()


def test_register_provider_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    isolated = ProviderRegistry()
    monkeypatch.setattr("src.providers.registry._default_registry", isolated)

    register_provider(MockProvider)

    assert isolated.exists("mock")


def test_cannot_register_abstract_class(registry: ProviderRegistry) -> None:
    from src.providers.base import BaseProvider

    with pytest.raises(TypeError, match="abstract"):
        registry.register(BaseProvider)
