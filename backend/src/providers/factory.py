"""Factory for constructing provider adapter instances from the registry."""

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
    """

    def __init__(self, registry: ProviderRegistry) -> None:
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
        """
        provider_cls = self._registry.get(provider_name)
        logger.info("Creating provider instance for %r", provider_name)
        return provider_cls(**kwargs)
