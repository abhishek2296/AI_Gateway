"""Unit tests for ModelRegistryService."""

from __future__ import annotations

import pytest

from src.registry.exceptions import AmbiguousModelError, ModelDisabledError, ModelNotFoundError
from src.registry.memory import MemoryModelRegistry
from src.registry.models import ModelCapability, ModelInfo, ProviderType
from src.services.model_registry_service import ModelRegistryService


async def _seed(service: ModelRegistryService, *models: ModelInfo) -> None:
    for model in models:
        await service.registry.register(model)


@pytest.mark.asyncio
async def test_resolve_default_model() -> None:
    service = ModelRegistryService(MemoryModelRegistry())
    await _seed(
        service,
        ModelInfo(
            name="qwen3:8b",
            provider=ProviderType.OLLAMA,
            display_name="Qwen3 8B",
            description="",
            capabilities=frozenset({ModelCapability.CHAT}),
            metadata={"is_default": True},
        ),
    )
    resolved = await service.resolve_for_chat()
    assert resolved.name == "qwen3:8b"


@pytest.mark.asyncio
async def test_resolve_disabled_raises() -> None:
    service = ModelRegistryService(MemoryModelRegistry())
    await _seed(
        service,
        ModelInfo(
            name="disabled-model",
            provider=ProviderType.OPENAI,
            display_name="Disabled",
            description="",
            capabilities=frozenset({ModelCapability.CHAT}),
            enabled=False,
        ),
    )
    with pytest.raises(ModelDisabledError):
        await service.resolve_for_chat(model="disabled-model", provider=ProviderType.OPENAI)


@pytest.mark.asyncio
async def test_resolve_ambiguous_name_raises() -> None:
    service = ModelRegistryService(MemoryModelRegistry())
    await _seed(
        service,
        ModelInfo(
            name="shared-name",
            provider=ProviderType.OPENAI,
            display_name="A",
            description="",
            capabilities=frozenset({ModelCapability.CHAT}),
        ),
        ModelInfo(
            name="shared-name",
            provider=ProviderType.OLLAMA,
            display_name="B",
            description="",
            capabilities=frozenset({ModelCapability.CHAT}),
        ),
    )
    with pytest.raises(AmbiguousModelError):
        await service.resolve_for_chat(model="shared-name")
