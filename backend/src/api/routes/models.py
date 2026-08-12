"""
HTTP routes exposing the gateway model catalog.

These endpoints read from :class:`~src.services.model_registry_service.ModelRegistryService`
and return the standard ``APIResponse[T]`` envelope used elsewhere in the API.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_model_registry_service
from src.core.enums import ProviderType
from src.core.exceptions import (
    AmbiguousModelHTTPException,
    ModelDisabledHTTPException,
    ModelNotFoundHTTPException,
)
from src.registry.exceptions import AmbiguousModelError, ModelDisabledError, ModelNotFoundError
from src.registry.models import ModelCapability
from src.schemas.common import APIResponse
from src.schemas.models import ModelListResponse, RegistryHealthResponse, RegistryModelResponse
from src.services.model_registry_service import ModelRegistryService

router = APIRouter(prefix="/models", tags=["models"])


def _map_registry_error(exc: Exception) -> None:
    if isinstance(exc, ModelNotFoundError):
        raise ModelNotFoundHTTPException(str(exc)) from exc
    if isinstance(exc, ModelDisabledError):
        raise ModelDisabledHTTPException(str(exc)) from exc
    if isinstance(exc, AmbiguousModelError):
        raise AmbiguousModelHTTPException(str(exc)) from exc
    raise exc


@router.get("/health", response_model=APIResponse[RegistryHealthResponse])
async def get_models_health(
    service: ModelRegistryService = Depends(get_model_registry_service),
) -> APIResponse[RegistryHealthResponse]:
    """
    Return a simple overview of model registry health.

    Counts and freshness are computed from the live catalog — not duplicated
    counters — so values stay accurate after startup refresh.
    """
    health = await service.get_registry_health()
    return APIResponse(data=RegistryHealthResponse(**health))


@router.get("", response_model=APIResponse[ModelListResponse])
async def list_models(
    provider: ProviderType | None = Query(default=None, description="Filter by provider family."),
    capability: ModelCapability | None = Query(default=None, description="Filter by capability."),
    enabled: bool | None = Query(default=None, description="Filter by enabled flag."),
    streaming: bool | None = Query(default=None, description="Filter by streaming support."),
    offset: int = Query(default=0, ge=0, description="Pagination offset."),
    limit: int | None = Query(default=None, ge=1, le=500, description="Pagination limit."),
    sort_by: str = Query(default="model_name", description="Sort field (model_name supported)."),
    service: ModelRegistryService = Depends(get_model_registry_service),
) -> APIResponse[ModelListResponse]:
    """
    List catalog models with optional filters and pagination.

    Filters combine with AND semantics. ``total`` reflects the count before
    ``offset``/``limit`` slicing so clients can paginate without extra calls.
    """
    models = await service.list_models(
        provider=provider,
        capability=capability,
        enabled=enabled,
        streaming=streaming,
    )

    if sort_by == "model_name":
        models = tuple(sorted(models, key=lambda m: m.name))

    total = len(models)
    page = models[offset:] if limit is None else models[offset : offset + limit]

    return APIResponse(
        data=ModelListResponse(
            items=[RegistryModelResponse.from_model_info(m) for m in page],
            total=total,
            offset=offset,
            limit=limit,
        ),
    )


@router.get("/{model_name}", response_model=APIResponse[RegistryModelResponse])
async def get_model_by_name(
    model_name: str,
    provider: ProviderType = Query(..., description="Provider family (required to disambiguate names)."),
    service: ModelRegistryService = Depends(get_model_registry_service),
) -> APIResponse[RegistryModelResponse]:
    """Fetch one model by provider and name."""
    try:
        model = await service.get_model(provider, model_name)
    except ModelNotFoundError as exc:
        _map_registry_error(exc)
    return APIResponse(data=RegistryModelResponse.from_model_info(model))


providers_router = APIRouter(prefix="/providers", tags=["models"])


@providers_router.get("/{provider}/models", response_model=APIResponse[ModelListResponse])
async def list_provider_models(
    provider: ProviderType,
    enabled: bool | None = Query(default=None),
    capability: ModelCapability | None = Query(default=None),
    streaming: bool | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1, le=500),
    service: ModelRegistryService = Depends(get_model_registry_service),
) -> APIResponse[ModelListResponse]:
    """List models offered by one provider family."""
    models = await service.list_models(
        provider=provider,
        enabled=enabled,
        capability=capability,
        streaming=streaming,
    )
    models = tuple(sorted(models, key=lambda m: m.name))
    total = len(models)
    page = models[offset:] if limit is None else models[offset : offset + limit]
    return APIResponse(
        data=ModelListResponse(
            items=[RegistryModelResponse.from_model_info(m) for m in page],
            total=total,
            offset=offset,
            limit=limit,
        ),
    )
