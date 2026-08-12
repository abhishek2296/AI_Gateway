"""Unit tests for model registry HTTP routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_model_registry_service
from src.api.routes.models import providers_router, router as models_router
from src.core.enums import ProviderType
from src.registry.memory import MemoryModelRegistry
from src.registry.models import ModelCapability, ModelInfo
from src.registry.read_only import ReadOnlyModelRegistryView
from src.services.model_registry_service import ModelRegistryService


@pytest.fixture
async def app() -> FastAPI:
    application = FastAPI()
    registry = MemoryModelRegistry()
    service = ModelRegistryService(ReadOnlyModelRegistryView(registry))
    await registry.register(
        ModelInfo(
            name="qwen3:8b",
            provider=ProviderType.OLLAMA,
            display_name="Qwen3 8B",
            description="local",
            capabilities=frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
        ),
    )

    application.include_router(models_router)
    application.include_router(providers_router)
    application.dependency_overrides[get_model_registry_service] = lambda: service
    return application


@pytest.mark.asyncio
async def test_list_models_endpoint(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/models", params={"provider": "ollama"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["total"] == 1


@pytest.mark.asyncio
async def test_models_health_endpoint(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/models/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["registered_models"] == 1
    assert payload["data"]["enabled_models"] == 1
    assert payload["data"]["default_model"] == "qwen3:8b"


@pytest.mark.asyncio
async def test_get_model_requires_provider_query(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/models/qwen3:8b",
            params={"provider": "ollama"},
        )
    assert response.status_code == 200
    assert response.json()["data"]["name"] == "qwen3:8b"
