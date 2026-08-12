"""
Read-only registry surface for normal runtime request handling.

During normal HTTP traffic the gateway must only **read** catalog entries.
Startup and future authenticated admin refresh jobs use
:class:`~src.registry.base.CatalogModelRegistry` for controlled writes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.core.enums import ProviderType
from src.registry.metrics import RegistryMetrics
from src.registry.models import ModelCapability, ModelInfo


class ReadOnlyModelRegistry(ABC):
    """
    Abstract read-only catalog contract exposed to routes and chat services.

    Why this interface exists:
        Phase 6.8 hardens the registry so request handlers cannot mutate the
        catalog accidentally. Writers (``ModelCatalogLoader``, future admin APIs)
        depend on :class:`~src.registry.base.CatalogModelRegistry` instead.
    """

    @abstractmethod
    async def get(self, provider: ProviderType, name: str) -> ModelInfo:
        """Fetch one model by provider and name."""

    @abstractmethod
    async def list(
        self,
        *,
        provider: ProviderType | None = None,
        capability: ModelCapability | None = None,
        enabled: bool | None = None,
        streaming: bool | None = None,
    ) -> Sequence[ModelInfo]:
        """Return catalog entries, optionally filtered (AND semantics)."""

    @abstractmethod
    async def exists(self, provider: ProviderType, name: str) -> bool:
        """Return whether ``(provider, name)`` is registered."""

    @abstractmethod
    async def providers(self) -> Sequence[ProviderType]:
        """Return distinct provider families present in the catalog."""

    @abstractmethod
    async def metrics(self, *, default_model: str | None = None) -> RegistryMetrics:
        """
        Compute catalog statistics from the current registry state.

        Args:
            default_model: Optional resolved default model name to include in
                the snapshot (typically supplied by :class:`ModelRegistryService`).

        Returns:
            A :class:`RegistryMetrics` instance with counts and refresh time.
        """


class ReadOnlyModelRegistryView(ReadOnlyModelRegistry):
    """
    Adapter that exposes only read methods from a writable catalog backend.

    Normal runtime code receives this wrapper so ``register`` / ``clear`` are
    not part of the dependency-injected interface, even though the underlying
    :class:`~src.registry.memory.MemoryModelRegistry` still supports writes for
    startup refresh and future admin tooling.

    Example:
        >>> writable = MemoryModelRegistry()
        >>> readonly = ReadOnlyModelRegistryView(writable)
        >>> hasattr(readonly, "register")
        False
    """

    def __init__(self, registry: ReadOnlyModelRegistry) -> None:
        """
        Wrap any object that already implements read + metrics methods.

        Args:
            registry: The live catalog backend (typically
                :class:`~src.registry.memory.MemoryModelRegistry`).
        """
        self._registry = registry

    async def get(self, provider: ProviderType, name: str) -> ModelInfo:
        return await self._registry.get(provider, name)

    async def list(
        self,
        *,
        provider: ProviderType | None = None,
        capability: ModelCapability | None = None,
        enabled: bool | None = None,
        streaming: bool | None = None,
    ) -> Sequence[ModelInfo]:
        return await self._registry.list(
            provider=provider,
            capability=capability,
            enabled=enabled,
            streaming=streaming,
        )

    async def exists(self, provider: ProviderType, name: str) -> bool:
        return await self._registry.exists(provider, name)

    async def providers(self) -> Sequence[ProviderType]:
        return await self._registry.providers()

    async def metrics(self, *, default_model: str | None = None) -> RegistryMetrics:
        return await self._registry.metrics(default_model=default_model)
