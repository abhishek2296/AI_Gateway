"""
Factory for constructing provider adapter instances from the registry.

``ProviderRegistry`` (see ``registry.py``) answers "what provider classes
exist?"; ``ProviderFactory`` answers the complementary question "give me a
working instance of one, configured for this request/tenant." Splitting
these two concerns means the registry can stay a simple, mostly-static
lookup table, while the factory can evolve independently to support things
like per-tenant credentials, connection pooling strategies, or test doubles
— without either concern's code needing to know about the other's
responsibilities.
"""

from __future__ import annotations

import logging
from typing import Any

from src.providers.base import BaseProvider
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Instantiate provider adapters on demand.

    Each call to :meth:`create` builds a new instance so callers receive an
    object with a clean lifecycle and request-scoped configuration.

    Building a fresh instance per call (rather than caching/reusing
    instances) keeps provider objects simple and stateless-by-default from
    the factory's point of view: each instance owns its own
    ``httpx.AsyncClient`` (see ``HTTPProviderMixin``) and any per-instance
    caches (e.g. ``ModelListCache``), so there's no risk of one caller's
    configuration (a different API key, base URL, or timeout) leaking into
    another caller's instance. Callers who do want to reuse a single
    long-lived instance (e.g. for a shared connection pool) are free to
    cache the object returned by :meth:`create` themselves.

    Attributes:
        _registry: The :class:`~src.providers.registry.ProviderRegistry`
            used to resolve a provider name to its class. Injected rather
            than using the module-level default registry directly, so a
            factory can be pointed at a test-specific registry with fake
            providers.

    Example:
        >>> from src.providers.base import BaseProvider
        >>> from src.providers.registry import ProviderRegistry
        >>> class DemoProvider(BaseProvider):
        ...     provider_name = "demo"
        ...     def __init__(self, greeting="hi"):
        ...         self.greeting = greeting
        ...     async def chat(self, request): ...
        ...     async def stream_chat(self, request): ...
        ...     async def embeddings(self, request): ...
        ...     async def list_models(self): return ()
        ...     async def health_check(self): ...
        >>> registry = ProviderRegistry()
        >>> registry.register(DemoProvider)
        >>> factory = ProviderFactory(registry)
        >>> provider = factory.create("demo", greeting="hello")
        >>> provider.greeting
        'hello'
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        """
        Bind this factory to the registry it will resolve provider names against.

        Args:
            registry: The provider registry to look up classes from. Pass
                ``get_registry()`` for the process-wide default, or a
                dedicated ``ProviderRegistry()`` instance in tests.
        """
        self._registry = registry

    def create(self, provider_name: str, **kwargs: Any) -> BaseProvider:
        """
        Resolve and instantiate the provider registered as ``provider_name``.

        Args:
            provider_name: Stable provider identifier (e.g. ``ollama``).
            **kwargs: Forwarded to the provider class constructor.

        Returns:
            A new provider instance.

        Raises:
            ProviderNotFoundError: If no provider is registered for the name.
            TypeError: If ``**kwargs`` does not match the resolved provider
                class's constructor signature (propagated from the
                underlying ``provider_cls(**kwargs)`` call, e.g. a missing
                required argument like ``api_key`` for a cloud provider).

        Example:
            >>> from src.providers.registry import get_registry
            >>> factory = ProviderFactory(get_registry())
            >>> provider = factory.create("ollama", base_url="http://localhost:11434")
            >>> provider.provider_name
            'ollama'
        """
        provider_cls = self._registry.get(provider_name)
        logger.info("Creating provider instance for %r", provider_name)
        return provider_cls(**kwargs)
