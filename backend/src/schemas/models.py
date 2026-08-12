"""
Pydantic schemas for model registry HTTP endpoints.

These DTOs translate registry :class:`~src.registry.models.ModelInfo` records
into API-friendly JSON while keeping the registry layer free of FastAPI imports.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.core.enums import ProviderType
from src.registry.models import ModelCapability, ModelInfo


class RegistryModelResponse(BaseModel):
    """
    One model entry returned by ``GET /models`` and related routes.

    Attributes:
        name: Provider-native model identifier used in chat requests.
        provider: Backend family serving this model.
        display_name: Human-friendly label for UIs.
        description: Free-text description (may be empty).
        capabilities: Advertised feature flags.
        context_window: Maximum input tokens, if known.
        max_output_tokens: Maximum output tokens, if known.
        enabled: Whether the model is currently routable.
        metadata: Additional key/value catalog metadata.
    """

    name: str
    provider: ProviderType
    display_name: str
    description: str
    capabilities: list[ModelCapability]
    context_window: int | None = None
    max_output_tokens: int | None = None
    enabled: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_model_info(cls, model: ModelInfo) -> RegistryModelResponse:
        """Build an API DTO from a registry catalog entry."""
        return cls(
            name=model.name,
            provider=model.provider,
            display_name=model.display_name,
            description=model.description,
            capabilities=sorted(model.capabilities, key=lambda c: c.value),
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            enabled=model.enabled,
            metadata=dict(model.metadata),
        )


class ModelListResponse(BaseModel):
    """
    Paginated-ready list payload for ``GET /models``.

    Attributes:
        items: Page of models after filters and optional slice.
        total: Count of models matching filters before pagination.
        offset: Applied offset (0 when pagination not used).
        limit: Applied limit, or ``null`` when all items returned.
    """

    items: list[RegistryModelResponse]
    total: int
    offset: int = 0
    limit: int | None = None


class RegistryHealthResponse(BaseModel):
    """
    Health overview for ``GET /models/health``.

    Attributes:
        status: Coarse registry state — ``healthy``, ``degraded``, or
            ``unhealthy``.
        registered_models: Total catalog entries.
        enabled_models: Routable models.
        disabled_models: Present but disabled entries.
        providers: Distinct provider families with at least one model.
        default_model: Resolved default chat model name, if any.
        last_refresh_time: ISO-8601 UTC timestamp of the last catalog refresh.
    """

    status: str
    registered_models: int
    enabled_models: int
    disabled_models: int
    providers: int
    default_model: str | None = None
    last_refresh_time: str | None = None
