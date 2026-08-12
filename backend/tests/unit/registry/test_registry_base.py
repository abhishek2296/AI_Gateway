"""
Unit tests for the abstract :class:`BaseModelRegistry` contract.

These tests do not require a production registry implementation (memory, Redis,
or database). They verify that the ABC cannot be used directly and that a
complete subclass satisfies the interface.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.enums import ProviderType
from src.registry.base import BaseModelRegistry
from src.registry.exceptions import ModelAlreadyRegisteredError, ModelNotFoundError
from src.registry.filters import ModelListFilters
from src.registry.metrics import RegistryMetrics
from src.registry.models import ModelCapability, ModelInfo


class _StubModelRegistry(BaseModelRegistry):
    """Minimal in-test double that stores models in a plain dict."""

    def __init__(self) -> None:
        self._models: dict[tuple[ProviderType, str], ModelInfo] = {}
        self._last_refresh_time: datetime | None = None

    def record_refresh(self, refreshed_at: datetime) -> None:
        self._last_refresh_time = refreshed_at

    async def metrics(self, *, default_model: str | None = None) -> RegistryMetrics:
        all_models = tuple(self._models.values())
        registered = len(all_models)
        enabled = sum(1 for model in all_models if model.enabled)
        return RegistryMetrics(
            registered_models=registered,
            enabled_models=enabled,
            disabled_models=registered - enabled,
            providers_count=len(dict.fromkeys(model.provider for model in all_models)),
            default_model=default_model,
            last_refresh_time=self._last_refresh_time,
        )

    async def register(self, model: ModelInfo) -> None:
        key = (model.provider, model.name)
        if key in self._models:
            raise ModelAlreadyRegisteredError(
                f"Model '{model.name}' is already registered for provider '{model.provider.value}'.",
                provider=model.provider,
                name=model.name,
            )
        self._models[key] = model

    async def unregister(self, provider: ProviderType, name: str) -> None:
        key = (provider, name)
        if key not in self._models:
            raise ModelNotFoundError(
                f"Model '{name}' is not registered for provider '{provider.value}'.",
                provider=provider,
                name=name,
            )
        del self._models[key]

    async def get(self, provider: ProviderType, name: str) -> ModelInfo:
        key = (provider, name)
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
    ) -> tuple[ModelInfo, ...]:
        filters = ModelListFilters.from_kwargs(
            provider=provider,
            capability=capability,
            enabled=enabled,
            streaming=streaming,
        )
        values = tuple(self._models.values())
        if filters is None:
            return values
        return tuple(model for model in values if filters.matches(model))

    async def exists(self, provider: ProviderType, name: str) -> bool:
        return (provider, name) in self._models

    async def providers(self) -> tuple[ProviderType, ...]:
        return tuple(dict.fromkeys(model.provider for model in self._models.values()))

    async def clear(self) -> None:
        self._models.clear()


def _sample_model(*, name: str = "gpt-4o", provider: ProviderType = ProviderType.OPENAI) -> ModelInfo:
    return ModelInfo(
        name=name,
        provider=provider,
        display_name="GPT-4o",
        description="Sample model for registry tests.",
        capabilities=frozenset({ModelCapability.CHAT}),
    )


def test_base_model_registry_cannot_be_instantiated_directly() -> None:
    """The ABC must stay abstract — callers need a concrete backend."""
    with pytest.raises(TypeError, match="abstract"):
        BaseModelRegistry()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_stub_registry_register_get_exists() -> None:
    """A complete subclass can register and retrieve a model by key."""
    registry = _StubModelRegistry()
    model = _sample_model()

    await registry.register(model)

    assert await registry.exists(ProviderType.OPENAI, "gpt-4o") is True
    assert await registry.get(ProviderType.OPENAI, "gpt-4o") == model


@pytest.mark.asyncio
async def test_stub_registry_list_and_providers() -> None:
    """list() returns all models; providers() deduplicates vendor families."""
    registry = _StubModelRegistry()
    await registry.register(_sample_model(name="gpt-4o", provider=ProviderType.OPENAI))
    await registry.register(_sample_model(name="qwen3:8b", provider=ProviderType.OLLAMA))

    models = await registry.list()
    providers = await registry.providers()

    assert len(models) == 2
    assert set(providers) == {ProviderType.OPENAI, ProviderType.OLLAMA}


@pytest.mark.asyncio
async def test_stub_registry_unregister_and_clear() -> None:
    """unregister removes one entry; clear removes everything."""
    registry = _StubModelRegistry()
    await registry.register(_sample_model())

    await registry.unregister(ProviderType.OPENAI, "gpt-4o")
    assert await registry.exists(ProviderType.OPENAI, "gpt-4o") is False

    await registry.register(_sample_model())
    await registry.clear()
    assert await registry.list() == ()


@pytest.mark.asyncio
async def test_stub_registry_duplicate_register_raises() -> None:
    """register must not silently overwrite an existing (provider, name) key."""
    registry = _StubModelRegistry()
    await registry.register(_sample_model())

    with pytest.raises(ModelAlreadyRegisteredError) as exc_info:
        await registry.register(_sample_model())

    assert exc_info.value.name == "gpt-4o"
    assert exc_info.value.provider is ProviderType.OPENAI


@pytest.mark.asyncio
async def test_stub_registry_get_missing_raises() -> None:
    """get on a missing model raises ModelNotFoundError with lookup context."""
    registry = _StubModelRegistry()

    with pytest.raises(ModelNotFoundError) as exc_info:
        await registry.get(ProviderType.GEMINI, "missing")

    assert exc_info.value.name == "missing"
    assert exc_info.value.provider is ProviderType.GEMINI
