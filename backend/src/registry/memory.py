"""
In-memory implementation of :class:`~src.registry.base.BaseModelRegistry`.

Used as the default catalog backend for development, tests, and single-process
deployments. A future Redis or database subclass can replace it without
changing routes or services because they depend on the abstract interface only.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from src.registry.base import BaseModelRegistry
from src.registry.exceptions import ModelAlreadyRegisteredError, ModelNotFoundError
from src.registry.filters import ModelListFilters
from src.registry.models import ModelCapability, ModelInfo, ProviderType


def _apply_list_filters(
    models: Sequence[ModelInfo],
    *,
    provider: ProviderType | None = None,
    capability: ModelCapability | None = None,
    enabled: bool | None = None,
    streaming: bool | None = None,
) -> tuple[ModelInfo, ...]:
    """
    Filter a sequence of models using shared :class:`ModelListFilters` logic.

    Kept as a module helper so memory, tests, and future backends apply
    identical filter semantics.
    """
    filters = ModelListFilters.from_kwargs(
        provider=provider,
        capability=capability,
        enabled=enabled,
        streaming=streaming,
    )
    if filters is None:
        return tuple(models)
    return tuple(model for model in models if filters.matches(model))


class MemoryModelRegistry(BaseModelRegistry):
    """
    Thread-safe, process-local model catalog backed by a plain dictionary.

    Why this class exists:
        The gateway needs a working registry before Redis (Phase 10) or a
        database-backed implementation is built. Memory storage is fast,
        requires no external services, and is ideal for unit tests.

    Model identity is ``(provider, name)``. All public methods acquire an
    ``RLock`` so concurrent HTTP requests and startup catalog loading cannot
    corrupt the internal dict.

    Example:
        >>> registry = MemoryModelRegistry()
        >>> await registry.register(some_model_info)
        >>> await registry.get(ProviderType.OPENAI, "gpt-4o")
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._models: dict[tuple[ProviderType, str], ModelInfo] = {}
        self._lock = threading.RLock()

    async def register(self, model: ModelInfo) -> None:
        """
        Store one model under ``(model.provider, model.name)``.

        Raises:
            ModelAlreadyRegisteredError: When the key already exists.
        """
        key = (model.provider, model.name)
        with self._lock:
            if key in self._models:
                raise ModelAlreadyRegisteredError(
                    f"Model '{model.name}' is already registered for provider '{model.provider.value}'.",
                    provider=model.provider,
                    name=model.name,
                )
            self._models[key] = model

    async def unregister(self, provider: ProviderType, name: str) -> None:
        """
        Remove one model from the catalog.

        Raises:
            ModelNotFoundError: When the key does not exist.
        """
        key = (provider, name)
        with self._lock:
            if key not in self._models:
                raise ModelNotFoundError(
                    f"Model '{name}' is not registered for provider '{provider.value}'.",
                    provider=provider,
                    name=name,
                )
            del self._models[key]

    async def get(self, provider: ProviderType, name: str) -> ModelInfo:
        """
        Fetch one model by provider and name.

        Raises:
            ModelNotFoundError: When the key does not exist.
        """
        key = (provider, name)
        with self._lock:
            if key not in self._models:
                raise ModelNotFoundError(
                    f"Model '{name}' is not registered for provider '{provider.value}'.",
                    provider=provider,
                    name=name,
                )
            return self._models[key]

    async def list(
        self,
        *,
        provider: ProviderType | None = None,
        capability: ModelCapability | None = None,
        enabled: bool | None = None,
        streaming: bool | None = None,
    ) -> Sequence[ModelInfo]:
        """Return all models, optionally filtered (AND semantics)."""
        with self._lock:
            values = tuple(self._models.values())
        return _apply_list_filters(
            values,
            provider=provider,
            capability=capability,
            enabled=enabled,
            streaming=streaming,
        )

    async def exists(self, provider: ProviderType, name: str) -> bool:
        """Return whether ``(provider, name)`` is registered."""
        with self._lock:
            return (provider, name) in self._models

    async def providers(self) -> Sequence[ProviderType]:
        """Return distinct provider families present in the catalog."""
        with self._lock:
            return tuple(dict.fromkeys(model.provider for model in self._models.values()))

    async def clear(self) -> None:
        """Remove every catalog entry."""
        with self._lock:
            self._models.clear()
