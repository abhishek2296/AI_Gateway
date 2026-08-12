"""
Runtime façade over :class:`~src.registry.read_only.ReadOnlyModelRegistry`.

Routes, chat, and catalog sync jobs depend on this service rather than the
raw registry so validation rules (enabled models, ambiguous names, defaults)
live in one place.
"""

from __future__ import annotations

import logging

from src.core.config import Settings, get_settings
from src.core.enums import ProviderType
from src.registry.exceptions import (
    AmbiguousModelError,
    ModelDisabledError,
    ModelNotFoundError,
)
from src.registry.metrics import RegistryMetrics
from src.registry.models import ModelCapability, ModelInfo
from src.registry.read_only import ReadOnlyModelRegistry

logger = logging.getLogger(__name__)


class ModelRegistryService:
    """
    High-level catalog operations for HTTP routes and ``AIService``.

    Wraps a :class:`ReadOnlyModelRegistry` backend with chat-specific resolution
    and filter forwarding so callers do not re-implement default-model logic.
    """

    def __init__(
        self,
        registry: ReadOnlyModelRegistry,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings or get_settings()

    @property
    def registry(self) -> ReadOnlyModelRegistry:
        """Underlying read-only registry view (memory, Redis, database, …)."""
        return self._registry

    async def get_model(self, provider: ProviderType, name: str) -> ModelInfo:
        """Fetch one model; propagates :class:`ModelNotFoundError` unchanged."""
        return await self._registry.get(provider, name)

    async def list_models(
        self,
        *,
        provider: ProviderType | None = None,
        capability: ModelCapability | None = None,
        enabled: bool | None = None,
        streaming: bool | None = None,
    ) -> tuple[ModelInfo, ...]:
        """Return catalog entries matching all supplied filters (AND semantics)."""
        result = await self._registry.list(
            provider=provider,
            capability=capability,
            enabled=enabled,
            streaming=streaming,
        )
        return tuple(result)

    async def get_metrics(self) -> RegistryMetrics:
        """
        Return catalog statistics derived from the live registry state.

        The default model name is resolved using the same rules as chat so
        health responses match runtime behavior.
        """
        default_name = await self._resolve_default_model_name()
        return await self._registry.metrics(default_model=default_name)

    async def get_registry_health(self) -> dict[str, object]:
        """
        Build a simple health overview for ``GET /models/health``.

        Returns:
            Dictionary with ``status``, model counts, provider count,
            ``default_model``, and ``last_refresh_time`` (ISO-8601 string or
            ``null`` when never refreshed).

        Example:
            >>> health = await service.get_registry_health()
            >>> health["status"] in {"healthy", "degraded", "unhealthy"}
            True
        """
        metrics = await self.get_metrics()
        status = self._derive_health_status(metrics)

        return {
            "status": status,
            "registered_models": metrics.registered_models,
            "enabled_models": metrics.enabled_models,
            "disabled_models": metrics.disabled_models,
            "providers": metrics.providers_count,
            "default_model": metrics.default_model,
            "last_refresh_time": (
                metrics.last_refresh_time.isoformat()
                if metrics.last_refresh_time is not None
                else None
            ),
        }

    @staticmethod
    def _derive_health_status(metrics: RegistryMetrics) -> str:
        """
        Map catalog metrics to a coarse health label.

        - ``unhealthy``: empty catalog or no enabled models (chat cannot route).
        - ``degraded``: models exist but startup refresh timestamp is missing.
        - ``healthy``: enabled models present and catalog was refreshed at startup.
        """
        if metrics.registered_models == 0 or metrics.enabled_models == 0:
            return "unhealthy"
        if metrics.last_refresh_time is None:
            return "degraded"
        return "healthy"

    async def resolve_for_chat(
        self,
        *,
        model: str | None = None,
        provider: ProviderType | str | None = None,
    ) -> ModelInfo:
        """
        Resolve which registry entry should handle a chat request.

        Registry-only resolution (Phase 6.7): every chat call must map to a
        catalog entry that exists and is enabled.

        Args:
            model: Optional explicit model name from the client.
            provider: Optional provider family. Required when ``model`` alone
                would match more than one provider.

        Returns:
            The selected :class:`ModelInfo`.

        Raises:
            ModelNotFoundError: When the requested or default model is absent.
            ModelDisabledError: When the model exists but ``enabled`` is false.
            AmbiguousModelError: When ``model`` matches multiple providers.
        """
        parsed_provider = self._parse_provider(provider) if provider is not None else None

        if model is not None:
            selected = await self._resolve_explicit_model(model, parsed_provider)
        else:
            selected = await self._resolve_default_model(parsed_provider)

        if not selected.enabled:
            raise ModelDisabledError(
                f"Model '{selected.name}' is disabled for provider '{selected.provider.value}'.",
                provider=selected.provider,
                name=selected.name,
            )

        logger.info(
            "Registry resolved chat model provider=%s model=%s enabled=%s",
            selected.provider.value,
            selected.name,
            selected.enabled,
        )
        return selected

    async def _resolve_default_model_name(self) -> str | None:
        """Return the default model name without raising when the catalog is empty."""
        try:
            resolved = await self._resolve_default_model(None)
        except ModelNotFoundError:
            return None
        return resolved.name

    async def _resolve_explicit_model(
        self,
        model: str,
        provider: ProviderType | None,
    ) -> ModelInfo:
        if provider is not None:
            return await self._registry.get(provider, model)

        matches = [entry for entry in await self._registry.list() if entry.name == model]
        if not matches:
            raise ModelNotFoundError(
                f"Model '{model}' is not registered in the catalog.",
                name=model,
            )
        if len(matches) > 1:
            raise AmbiguousModelError(
                f"Model name '{model}' is registered for multiple providers; specify ?provider=.",
                name=model,
            )
        return matches[0]

    async def _resolve_default_model(self, provider: ProviderType | None) -> ModelInfo:
        candidates = await self._registry.list(provider=provider, enabled=True)

        for entry in candidates:
            if entry.metadata.get("is_default") is True:
                return entry

        default_provider = provider or ProviderType(self._settings.DEFAULT_PROVIDER)
        default_name = self._settings.DEFAULT_MODEL
        if await self._registry.exists(default_provider, default_name):
            return await self._registry.get(default_provider, default_name)

        if candidates:
            return candidates[0]

        raise ModelNotFoundError(
            "No enabled models are registered in the catalog.",
            provider=default_provider,
            name=default_name,
        )

    @staticmethod
    def _parse_provider(provider: ProviderType | str) -> ProviderType:
        if isinstance(provider, ProviderType):
            return provider
        return ProviderType(provider)
