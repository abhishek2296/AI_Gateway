"""
Thread-safe registry of :class:`~src.providers.base.BaseProvider` implementations.

The registry is the mechanism by which the gateway discovers which provider
adapters exist without any layer above it (routes, services) needing to
import concrete adapter classes directly. Each vendor adapter module
(``ollama.py``, ``openai.py``, ``anthropic.py``, ``gemini.py``) calls
:func:`register_provider` at import time (see the bottom of each of those
modules), so simply importing ``src.providers`` (which imports every adapter
module — see ``__init__.py``) populates the process-wide default registry
returned by :func:`get_registry`.

This indirection is what lets ``05-ai-gateway-architecture.mdc``'s goal of
"adding a provider requires one service class + config + DI registration, no
route changes" hold true: the registry maps a stable string name (e.g.
``"openai"``) to a provider *class*; instantiating it with the right
credentials/config is a separate concern handled by
:class:`~src.providers.factory.ProviderFactory`.
"""

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

    Deliberately stores *classes* rather than pre-built instances: a
    provider instance carries request-scoped/expensive state (an
    ``httpx.AsyncClient``, credentials, per-instance caches), whereas the
    registry's job is only to answer "what provider classes exist and what
    are they called?" — a question that doesn't need a live client or
    credentials at all. This also means multiple instances of the same
    provider (e.g. with different API keys for a multi-tenant deployment)
    can be created from a single registration.

    Attributes:
        _providers: Internal mapping of stable provider name (e.g.
            ``"ollama"``) to the registered provider class. Not part of the
            public interface — use :meth:`get`/:meth:`exists`/:meth:`available`
            instead of reading this directly.
        _lock: A re-entrant lock (``threading.RLock``) guarding all reads and
            writes to ``_providers``, so registration/lookup is safe even if
            called concurrently from multiple threads (e.g. a sync
            "register on import" call happening while the async event loop
            is elsewhere handling requests that call :meth:`get`).
            Re-entrant (as opposed to a plain ``Lock``) so a method that
            already holds the lock (none currently do) could safely call
            another locking method on the same registry without deadlocking.

    Example:
        >>> from src.providers.base import BaseProvider
        >>> class DemoProvider(BaseProvider):
        ...     provider_name = "demo"
        ...     async def chat(self, request): ...
        ...     async def stream_chat(self, request): ...
        ...     async def embeddings(self, request): ...
        ...     async def list_models(self): return ()
        ...     async def health_check(self): ...
        >>> registry = ProviderRegistry()
        >>> registry.register(DemoProvider)
        >>> registry.exists("demo")
        True
        >>> registry.get("demo") is DemoProvider
        True
    """

    def __init__(self) -> None:
        """Create an empty registry with no providers registered yet."""
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

        Example:
            >>> from src.providers.base import BaseProvider
            >>> class DemoProvider(BaseProvider):
            ...     provider_name = "demo"
            ...     async def chat(self, request): ...
            ...     async def stream_chat(self, request): ...
            ...     async def embeddings(self, request): ...
            ...     async def list_models(self): return ()
            ...     async def health_check(self): ...
            >>> registry = ProviderRegistry()
            >>> registry.register(DemoProvider)
            >>> registry.register(DemoProvider)
            Traceback (most recent call last):
                ...
            ProviderError: Provider 'demo' is already registered.
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

        Example:
            >>> from src.providers.base import BaseProvider
            >>> class DemoProvider(BaseProvider):
            ...     provider_name = "demo"
            ...     async def chat(self, request): ...
            ...     async def stream_chat(self, request): ...
            ...     async def embeddings(self, request): ...
            ...     async def list_models(self): return ()
            ...     async def health_check(self): ...
            >>> registry = ProviderRegistry()
            >>> registry.register(DemoProvider)
            >>> registry.unregister("demo")
            >>> registry.exists("demo")
            False
            >>> registry.unregister("demo")
            Traceback (most recent call last):
                ...
            ProviderNotFoundError: Provider 'demo' is not registered.
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

        Example:
            >>> registry = ProviderRegistry()
            >>> registry.get("nonexistent")
            Traceback (most recent call last):
                ...
            ProviderNotFoundError: Provider 'nonexistent' is not registered.
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
        """
        Return whether ``name`` is registered.

        Args:
            name: The provider name to check, e.g. ``"openai"``.

        Returns:
            ``True`` if a provider class is registered under ``name``,
            ``False`` otherwise. Provided as a non-raising alternative to
            :meth:`get` for call sites that want to branch on availability
            rather than handle an exception.

        Example:
            >>> registry = ProviderRegistry()
            >>> registry.exists("openai")
            False
        """
        with self._lock:
            return name in self._providers

    def available(self) -> tuple[str, ...]:
        """
        Return registered provider names in sorted order.

        Returns:
            A tuple of every currently registered provider name, sorted
            alphabetically. Sorted (rather than insertion order) so the
            output is deterministic and stable across processes/restarts
            regardless of module import order — useful for things like a
            ``GET /providers`` listing endpoint or log output.

        Example:
            >>> from src.providers.base import BaseProvider
            >>> class DemoProvider(BaseProvider):
            ...     provider_name = "demo"
            ...     async def chat(self, request): ...
            ...     async def stream_chat(self, request): ...
            ...     async def embeddings(self, request): ...
            ...     async def list_models(self): return ()
            ...     async def health_check(self): ...
            >>> registry = ProviderRegistry()
            >>> registry.register(DemoProvider)
            >>> registry.available()
            ('demo',)
        """
        with self._lock:
            return tuple(sorted(self._providers))

    def clear(self) -> None:
        """
        Remove every registered provider class.

        Primarily useful in tests, to reset registry state between test
        cases so registrations from one test don't leak into another
        (registries are otherwise long-lived/process-wide via
        :func:`get_registry`).

        Returns:
            None.

        Example:
            >>> from src.providers.base import BaseProvider
            >>> class DemoProvider(BaseProvider):
            ...     provider_name = "demo"
            ...     async def chat(self, request): ...
            ...     async def stream_chat(self, request): ...
            ...     async def embeddings(self, request): ...
            ...     async def list_models(self): return ()
            ...     async def health_check(self): ...
            >>> registry = ProviderRegistry()
            >>> registry.register(DemoProvider)
            >>> registry.clear()
            >>> registry.available()
            ()
        """
        with self._lock:
            count = len(self._providers)
            self._providers.clear()
            logger.info("Cleared provider registry (%d entries removed)", count)

    @staticmethod
    def _validate_provider_class(provider_cls: type[BaseProvider]) -> None:
        """
        Guard :meth:`register` against non-class, non-``BaseProvider``, or abstract input.

        Two distinct failure modes are checked because they indicate
        different mistakes: passing something that isn't a
        ``BaseProvider`` subclass at all (a wiring bug), versus passing the
        abstract ``BaseProvider`` itself or a subclass that hasn't
        implemented every ``@abstractmethod`` (an incomplete adapter). Both
        are caught here, before ``_resolve_provider_name`` even runs, so the
        error clearly names the real problem instead of surfacing as a
        confusing failure later when someone tries to instantiate the
        "registered" class.

        Args:
            provider_cls: The candidate class passed to :meth:`register`.

        Returns:
            None. Raises on any validation failure; returns nothing on
            success.

        Raises:
            TypeError: If ``provider_cls`` is not a class, is not a subclass
                of :class:`~src.providers.base.BaseProvider`, or is abstract
                (i.e. does not implement every required abstract method).

        Example:
            >>> ProviderRegistry._validate_provider_class(object)
            Traceback (most recent call last):
                ...
            TypeError: Expected a BaseProvider subclass, got <class 'object'>.
        """
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

    Deliberately reads ``provider_cls.__dict__`` rather than using
    ``getattr(provider_cls, "provider_name")``: the former only finds a
    ``provider_name`` defined directly on ``provider_cls`` itself, whereas
    ``getattr`` would also find one inherited from a parent class. Since
    ``BaseProvider`` itself intentionally declares ``provider_name`` as an
    abstract *property* rather than a plain attribute, this distinction
    mainly guards against a future subclass forgetting to define its own
    ``provider_name`` and silently inheriting an unrelated one.

    Args:
        provider_cls: The provider class to read the name from.

    Returns:
        The non-empty string found on ``provider_cls.provider_name``.

    Raises:
        TypeError: If ``provider_cls`` does not define a non-empty string
            class attribute named ``provider_name``.

    Example:
        >>> from src.providers.base import BaseProvider
        >>> class DemoProvider(BaseProvider):
        ...     provider_name = "demo"
        ...     async def chat(self, request): ...
        ...     async def stream_chat(self, request): ...
        ...     async def embeddings(self, request): ...
        ...     async def list_models(self): return ()
        ...     async def health_check(self): ...
        >>> _resolve_provider_name(DemoProvider)
        'demo'
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
    """
    Return the process-wide default :class:`ProviderRegistry`.

    This module-level singleton is what every adapter module's
    ``register_provider(...)`` call (at import time) and every service-layer
    lookup ultimately share, so the gateway has a single, consistent view of
    "which providers are available" without needing dependency injection to
    thread a registry instance through every layer.

    Returns:
        The single, shared ``ProviderRegistry`` instance for this process.
        Always returns the same object across calls.

    Example:
        >>> get_registry() is get_registry()
        True
    """
    return _default_registry


def register_provider(provider_cls: type[BaseProvider]) -> None:
    """
    Register ``provider_cls`` on the default registry.

    Provider modules call this at import time to enable automatic discovery.

    This module-level convenience function (as opposed to requiring every
    adapter module to call ``get_registry().register(...)`` explicitly) is
    what each vendor adapter module invokes as its very last line (e.g.
    ``register_provider(OllamaProvider)`` at the bottom of ``ollama.py``), so
    simply importing that module has the side effect of making the provider
    discoverable.

    Args:
        provider_cls: The concrete provider class to register, per the same
            rules as :meth:`ProviderRegistry.register`.

    Raises:
        TypeError: If ``provider_cls`` is not a concrete ``BaseProvider``
            subclass.
        ProviderError: If a provider with the same ``provider_name`` is
            already registered (e.g. if the module were accidentally
            imported twice under different module names).

    Example:
        >>> from src.providers.base import BaseProvider
        >>> class DemoProvider(BaseProvider):
        ...     provider_name = "demo-example"
        ...     async def chat(self, request): ...
        ...     async def stream_chat(self, request): ...
        ...     async def embeddings(self, request): ...
        ...     async def list_models(self): return ()
        ...     async def health_check(self): ...
        >>> register_provider(DemoProvider)
        >>> get_registry().exists("demo-example")
        True
    """
    _default_registry.register(provider_cls)
