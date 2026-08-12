"""Unit tests for registry metrics and read-only runtime surface."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.enums import ProviderType
from src.registry.memory import MemoryModelRegistry
from src.registry.models import ModelCapability, ModelInfo
from src.registry.read_only import ReadOnlyModelRegistryView
from src.services.model_registry_service import ModelRegistryService


def _sample_model(*, name: str = "gpt-4o", enabled: bool = True) -> ModelInfo:
    return ModelInfo(
        name=name,
        provider=ProviderType.OPENAI,
        display_name="GPT-4o",
        description="Sample",
        capabilities=frozenset({ModelCapability.CHAT}),
        enabled=enabled,
    )


@pytest.mark.asyncio
async def test_provider_type_is_single_shared_enum() -> None:
    """Registry and API layers must reference the same ProviderType class."""
    from src.core.enums import ProviderType as CoreProviderType
    from src.registry import ProviderType as RegistryPackageProviderType
    from src.registry.models import ProviderType as ModelsProviderType

    assert CoreProviderType is RegistryPackageProviderType
    assert CoreProviderType is ModelsProviderType
    assert CoreProviderType.OLLAMA == "ollama"


@pytest.mark.asyncio
async def test_metrics_count_enabled_and_disabled_models() -> None:
    registry = MemoryModelRegistry()
    await registry.register(_sample_model(name="enabled-model", enabled=True))
    await registry.register(_sample_model(name="disabled-model", enabled=False))

    metrics = await registry.metrics()

    assert metrics.registered_models == 2
    assert metrics.enabled_models == 1
    assert metrics.disabled_models == 1
    assert metrics.providers_count == 1


@pytest.mark.asyncio
async def test_metrics_update_after_refresh() -> None:
    registry = MemoryModelRegistry()
    refreshed_at = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
    registry.record_refresh(refreshed_at)
    await registry.register(_sample_model())

    metrics = await registry.metrics(default_model="gpt-4o")

    assert metrics.last_refresh_time == refreshed_at
    assert metrics.default_model == "gpt-4o"


@pytest.mark.asyncio
async def test_readonly_view_hides_write_methods() -> None:
    writable = MemoryModelRegistry()
    readonly = ReadOnlyModelRegistryView(writable)

    assert not hasattr(readonly, "register")
    assert not hasattr(readonly, "unregister")
    assert not hasattr(readonly, "clear")


@pytest.mark.asyncio
async def test_service_health_unhealthy_when_catalog_empty() -> None:
    registry = MemoryModelRegistry()
    service = ModelRegistryService(ReadOnlyModelRegistryView(registry))

    health = await service.get_registry_health()

    assert health["status"] == "unhealthy"
    assert health["registered_models"] == 0


@pytest.mark.asyncio
async def test_service_health_degraded_without_refresh_timestamp() -> None:
    registry = MemoryModelRegistry()
    await registry.register(_sample_model())
    service = ModelRegistryService(ReadOnlyModelRegistryView(registry))

    health = await service.get_registry_health()

    assert health["status"] == "degraded"
    assert health["enabled_models"] == 1


@pytest.mark.asyncio
async def test_service_health_healthy_after_refresh() -> None:
    registry = MemoryModelRegistry()
    registry.record_refresh(datetime.now(UTC))
    await registry.register(_sample_model())
    service = ModelRegistryService(ReadOnlyModelRegistryView(registry))

    health = await service.get_registry_health()

    assert health["status"] == "healthy"
    assert health["default_model"] == "gpt-4o"
