"""Unit tests for in-memory model registry."""

from __future__ import annotations

import threading

import pytest

from src.registry.exceptions import ModelAlreadyRegisteredError, ModelNotFoundError
from src.registry.memory import MemoryModelRegistry
from src.registry.models import ModelCapability, ModelInfo, ProviderType


def _model(
    *,
    name: str = "gpt-4o",
    provider: ProviderType = ProviderType.OPENAI,
    capabilities: frozenset[ModelCapability] | None = None,
    enabled: bool = True,
) -> ModelInfo:
    return ModelInfo(
        name=name,
        provider=provider,
        display_name=name,
        description="test",
        capabilities=capabilities or frozenset({ModelCapability.CHAT}),
        enabled=enabled,
    )


@pytest.mark.asyncio
async def test_memory_registry_register_and_get() -> None:
    registry = MemoryModelRegistry()
    model = _model()
    await registry.register(model)
    assert await registry.get(ProviderType.OPENAI, "gpt-4o") == model


@pytest.mark.asyncio
async def test_memory_registry_duplicate_raises() -> None:
    registry = MemoryModelRegistry()
    await registry.register(_model())
    with pytest.raises(ModelAlreadyRegisteredError):
        await registry.register(_model())


@pytest.mark.asyncio
async def test_memory_registry_list_filters() -> None:
    registry = MemoryModelRegistry()
    await registry.register(
        _model(
            name="qwen3:8b",
            provider=ProviderType.OLLAMA,
            capabilities=frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
        ),
    )
    await registry.register(
        _model(
            name="gpt-4o",
            provider=ProviderType.OPENAI,
            capabilities=frozenset({ModelCapability.CHAT}),
            enabled=False,
        ),
    )

    ollama_enabled = await registry.list(provider=ProviderType.OLLAMA, enabled=True)
    assert len(ollama_enabled) == 1

    streaming = await registry.list(streaming=True)
    assert len(streaming) == 1
    assert streaming[0].name == "qwen3:8b"


@pytest.mark.asyncio
async def test_memory_registry_thread_safe_register() -> None:
    registry = MemoryModelRegistry()
    errors: list[Exception] = []

    def worker(index: int) -> None:
        import asyncio

        async def run() -> None:
            try:
                await registry.register(
                    _model(name=f"model-{index}", provider=ProviderType.OLLAMA),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        asyncio.run(run())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(await registry.list()) == 20


@pytest.mark.asyncio
async def test_memory_registry_unregister_missing_raises() -> None:
    registry = MemoryModelRegistry()
    with pytest.raises(ModelNotFoundError):
        await registry.unregister(ProviderType.GEMINI, "missing")
