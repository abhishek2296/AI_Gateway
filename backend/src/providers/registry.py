"""Thread-safe registry of :class:`~src.providers.base.BaseProvider` implementations."""

from __future__ import annotations

import inspect
import logging
import threading
from typing import TypeVar

from src.providers.base import BaseProvider
from src.providers.exceptions import ProviderError, ProviderNotFoundError

logger = logging.getLogger(__name__)

ProviderType = TypeVar("ProviderType", bound=type[BaseProvider])


class ProviderRegistry:
    """
    Maps provider names to provider **classes** (not instances).

    The registry is thread-safe and intended to be shared across requests.
    Instantiation is delegated to :class:`~src.providers.factory.ProviderFactory`.
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseProvider]] = {}
        self._lock = threading.RLock()

    def register(self, provider_cls: type[BaseProvider]) -> None:
        """
        Register a concrete provider class under its ``provider_name``.

        Args:
            provider_cls: A non-abstract subclass of :class:`BaseProvider` that
                defines a string class attribute ``provider_name``.

        Raises:
            TypeError: If ``provider_cls`` is not a concrete ``BaseProvider``.
            ProviderError: If the provider name is already registered.
        """
        self._validate_provider_class(provider_cls)
        name = _resolve_provider_name(provider_cls)

        with self._lock:
            if name in self._providers:
                logger.warning(
                    "Duplicate provider registration attempted for %r",
                    name,
                )
                raise ProviderError(
                    f"Provider {name!r} is already registered.",
                    provider=name,
                )
            self._providers[name] = provider_cls
            logger.info("Registered provider class %s as %r", provider_cls.__name__, name)

    def unregister(self, name: str) -> None:
        """
        Remove a provider class from the registry.

        Raises:
            ProviderNotFoundError: If ``name`` is not registered.
        """
        with self._lock:
            if name not in self._providers:
                logger.warning("Unregister failed; provider %r not found", name)
                raise ProviderNotFoundError(
                    f"Provider {name!r} is not registered.",
                    provider=name,
                )
            del self._providers[name]
            logger.info("Unregistered provider %r", name)

    def get(self, name: str) -> type[BaseProvider]:
        """
        Return the registered provider class for ``name``.

        Raises:
            ProviderNotFoundError: If ``name`` is not registered.
        """
        with self._lock:
            provider_cls = self._providers.get(name)
            if provider_cls is None:
                logger.warning("Provider lookup failed for %r", name)
                raise ProviderNotFoundError(
                    f"Provider {name!r} is not registered.",
                    provider=name,
                )
            return provider_cls

    def exists(self, name: str) -> bool:
        """Return whether ``name`` is registered."""
        with self._lock:
            return name in self._providers

    def available(self) -> tuple[str, ...]:
        """Return registered provider names in sorted order."""
        with self._lock:
            return tuple(sorted(self._providers))

    def clear(self) -> None:
        """Remove every registered provider class."""
        with self._lock:
            count = len(self._providers)
            self._providers.clear()
            logger.info("Cleared provider registry (%d entries removed)", count)

    @staticmethod
    def _validate_provider_class(provider_cls: type[BaseProvider]) -> None:
        if not isinstance(provider_cls, type) or not issubclass(provider_cls, BaseProvider):
            raise TypeError(
                f"Expected a BaseProvider subclass, got {provider_cls!r}.",
            )
        if inspect.isabstract(provider_cls):
            raise TypeError(
                f"Cannot register abstract provider class {provider_cls.__name__}.",
            )


def _resolve_provider_name(provider_cls: type[BaseProvider]) -> str:
    """
    Read the registry key from a provider class.

    Concrete providers must define ``provider_name`` as a string class attribute.
    """
    name = provider_cls.__dict__.get("provider_name")
    if isinstance(name, str) and name:
        return name
    raise TypeError(
        f"{provider_cls.__name__} must define a non-empty string class attribute "
        f"'provider_name' to be registered.",
    )


_default_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Return the process-wide default :class:`ProviderRegistry`."""
    return _default_registry


def register_provider(provider_cls: type[BaseProvider]) -> None:
    """
    Register ``provider_cls`` on the default registry.

    Provider modules call this at import time to enable automatic discovery.
    """
    _default_registry.register(provider_cls)
