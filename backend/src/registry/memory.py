"""
In-memory implementation of :class:`~src.registry.base.CatalogModelRegistry`.

Used as the default catalog backend for development, tests, and single-process
deployments. A future Redis or database subclass can replace it without
changing routes or services because they depend on the abstract interface only.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import datetime

from src.core.enums import ProviderType
from src.registry.base import CatalogModelRegistry
from src.registry.exceptions import ModelAlreadyRegisteredError, ModelNotFoundError
from src.registry.filters import ModelListFilters
from src.registry.metrics import RegistryMetrics
from src.registry.models import ModelCapability, ModelInfo


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


class MemoryModelRegistry(CatalogModelRegistry):
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
        self._last_refresh_time: datetime | None = None

    def record_refresh(self, refreshed_at: datetime) -> None:
        """
        Remember when the catalog was last repopulated.

        Args:
            refreshed_at: Timezone-aware UTC timestamp from the loader.
        """
        with self._lock:
            self._last_refresh_time = refreshed_at

    async def metrics(self, *, default_model: str | None = None) -> RegistryMetrics:
        """
        Compute catalog statistics from the current in-memory contents.

        We count models on each call instead of storing separate counters so
        numbers stay accurate after ``clear()`` or ``register()`` during refresh.
        """
        with self._lock:
            all_models = tuple(self._models.values())
            last_refresh = self._last_refresh_time

        registered = len(all_models)
        enabled = sum(1 for model in all_models if model.enabled)
        disabled = registered - enabled
        providers_count = len(dict.fromkeys(model.provider for model in all_models))

        return RegistryMetrics(
            registered_models=registered,
            enabled_models=enabled,
            disabled_models=disabled,
            providers_count=providers_count,
            default_model=default_model,
            last_refresh_time=last_refresh,
        )

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
