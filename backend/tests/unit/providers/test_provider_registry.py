"""
Unit tests for ``ProviderRegistry`` and ``ProviderFactory``.

``ProviderRegistry`` (``src/providers/registry.py``) maps provider name
strings to provider *classes*, and ``ProviderFactory``
(``src/providers/factory.py``) turns a registered class into a live
*instance*. These tests use ``MockProvider`` (see ``mock_provider.py``)
instead of a real vendor provider so registration/lookup/instantiation
behavior can be verified without any network dependency.
"""

from __future__ import annotations

import pytest

from src.providers.exceptions import ProviderError, ProviderNotFoundError
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry, register_provider
from tests.unit.providers.mock_provider import MockProvider


@pytest.fixture
def registry() -> ProviderRegistry:
    """Return a fresh, empty ``ProviderRegistry`` isolated from the process-wide default.

    A brand-new instance per test prevents state (e.g. previously registered
    provider names) from leaking between tests, since the module also
    exposes a shared, mutable default registry.
    """
    return ProviderRegistry()


@pytest.fixture
def factory(registry: ProviderRegistry) -> ProviderFactory:
    """Return a ``ProviderFactory`` bound to the per-test isolated ``registry`` fixture."""
    return ProviderFactory(registry)


def test_successful_registration(registry: ProviderRegistry) -> None:
    """
    Registering a concrete provider class makes it discoverable by name.

    Confirms the three read paths (`exists`, `get`, `available`) all agree
    once `MockProvider` has been registered under its `provider_name`.
    """
    registry.register(MockProvider)

    assert registry.exists("mock")
    assert registry.get("mock") is MockProvider
    assert registry.available() == ("mock",)


def test_duplicate_registration_raises(registry: ProviderRegistry) -> None:
    """
    Registering the same provider name twice must fail loudly.

    Silently allowing a second registration to overwrite the first could mask
    a bug where two provider modules accidentally share a `provider_name`.
    """
    registry.register(MockProvider)

    with pytest.raises(ProviderError, match="already registered"):
        registry.register(MockProvider)


def test_unregister(registry: ProviderRegistry) -> None:
    """Unregistering a provider removes it from both `exists()` and `available()`."""
    registry.register(MockProvider)
    registry.unregister("mock")

    assert not registry.exists("mock")
    assert registry.available() == ()


def test_unregister_missing_raises(registry: ProviderRegistry) -> None:
    """Unregistering a name that was never registered raises `ProviderNotFoundError`."""
    with pytest.raises(ProviderNotFoundError):
        registry.unregister("missing")


def test_create_instantiates_provider(
    registry: ProviderRegistry,
    factory: ProviderFactory,
) -> None:
    """
    The factory instantiates the registered class and forwards constructor kwargs.

    Confirms `factory.create(name, **kwargs)` resolves the class from the
    registry and passes `endpoint="http://test"` straight through to
    `MockProvider.__init__`, which is how callers configure a provider
    instance (e.g. with request-scoped credentials or endpoints).
    """
    registry.register(MockProvider)

    provider = factory.create("mock", endpoint="http://test")

    assert isinstance(provider, MockProvider)
    assert provider.endpoint == "http://test"


def test_create_returns_fresh_instances(registry: ProviderRegistry, factory: ProviderFactory) -> None:
    """
    Each call to `create()` returns a new instance, not a shared singleton.

    Provider instances may hold request-scoped state (e.g. an HTTP client),
    so reusing the same instance across unrelated calls would risk state
    leaking between callers.
    """
    registry.register(MockProvider)

    first = factory.create("mock")
    second = factory.create("mock")

    assert first is not second


def test_create_missing_provider_raises(factory: ProviderFactory) -> None:
    """Creating an unregistered provider name raises `ProviderNotFoundError`."""
    with pytest.raises(ProviderNotFoundError):
        factory.create("missing")


def test_get_missing_raises(registry: ProviderRegistry) -> None:
    """Looking up an unregistered provider name raises `ProviderNotFoundError`."""
    with pytest.raises(ProviderNotFoundError):
        registry.get("missing")


def test_available_sorted(registry: ProviderRegistry) -> None:
    """
    `available()` returns provider names sorted alphabetically, not insertion order.

    Deterministic ordering matters for any caller that surfaces this list
    (e.g. an API response or CLI listing) so output doesn't depend on
    module import order.
    """

    class AlphaProvider(MockProvider):
        provider_name = "alpha"

    class ZetaProvider(MockProvider):
        provider_name = "zeta"

    # Registered out of alphabetical order (zeta before alpha) to prove the
    # registry itself sorts the result rather than relying on insertion order.
    registry.register(ZetaProvider)
    registry.register(AlphaProvider)

    assert registry.available() == ("alpha", "zeta")


def test_clear(registry: ProviderRegistry) -> None:
    """`clear()` removes every registered provider, leaving `available()` empty."""
    registry.register(MockProvider)
    registry.clear()

    assert registry.available() == ()


def test_register_provider_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The module-level `register_provider()` helper writes to the shared default registry.

    Provider modules call this helper at import time for automatic
    discovery, so it must delegate to the process-wide default registry
    rather than requiring an explicit registry argument. The default is
    monkeypatched with an isolated instance so this test doesn't pollute
    global state shared with other tests.
    """
    isolated = ProviderRegistry()
    monkeypatch.setattr("src.providers.registry._default_registry", isolated)

    register_provider(MockProvider)

    assert isolated.exists("mock")


def test_cannot_register_abstract_class(registry: ProviderRegistry) -> None:
    """
    Registering `BaseProvider` itself (an abstract class) must raise `TypeError`.

    `BaseProvider` has no usable implementation of its abstract methods;
    allowing it into the registry would let a factory "successfully"
    instantiate a provider that immediately fails on every call.
    """
    from src.providers.base import BaseProvider

    with pytest.raises(TypeError, match="abstract"):
        registry.register(BaseProvider)
