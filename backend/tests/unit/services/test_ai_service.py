"""Unit tests for AIService using MockProvider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import Settings
from src.core.enums import ProviderType
from src.core.exceptions import LLMResponseException, OllamaConnectionException
from src.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthCheckResult,
    ModelInfo,
)
from src.providers.exceptions import ModelNotFoundError, ProviderUnavailableError
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry
from src.services.ai_service import AIService
from src.services.resolution_types import ResolutionSource, ResolvedConfiguration


class StubProvider(BaseProvider):
    """Test double registered under the ollama provider name."""

    provider_name = "ollama"

    def __init__(self) -> None:
        self.chat_calls: list[ChatRequest] = []
        self.health_calls = 0
        self.closed = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_calls.append(request)
        return ChatResponse(content="stub-response", model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="stub")

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])

    async def list_models(self) -> Sequence[ModelInfo]:
        return (ModelInfo(id="stub", name="stub"),)

    async def health_check(self) -> HealthCheckResult:
        self.health_calls += 1
        return HealthCheckResult(healthy=True, latency_ms=12.5)

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> StubProvider:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()


class ClosableOnlyProvider(BaseProvider):
    """Provider with explicit close() only — no async context manager."""

    provider_name = "ollama"

    def __init__(self) -> None:
        self.closed = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="closable-response", model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="stub")

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])

    async def list_models(self) -> Sequence[ModelInfo]:
        return (ModelInfo(id="stub", name="stub"),)

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, latency_ms=1.0)

    async def close(self) -> None:
        self.closed = True


def _settings(**overrides: object) -> Settings:
    defaults = {
        "DATABASE_URL": "postgresql+asyncpg://ai_app:ai_app_password@localhost:5432/ai_coding_assistant",
        "DEFAULT_PROVIDER": "ollama",
        "DEFAULT_MODEL": "qwen3:8b",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _resolved_config(
    *,
    provider_name: str = "ollama",
    model_name: str = "qwen3:8b",
    factory_kwargs: dict[str, object] | None = None,
) -> ResolvedConfiguration:
    return ResolvedConfiguration(
        provider_name=provider_name,
        model_name=model_name,
        factory_kwargs=factory_kwargs or {"base_url": "http://localhost:11434", "timeout": 60.0},
        provider_source=ResolutionSource.SETTINGS,
        model_source=ResolutionSource.SETTINGS,
    )


@pytest.fixture
def stub_provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def resolution_coordinator() -> AsyncMock:
    coordinator = AsyncMock()
    coordinator.resolve = AsyncMock(return_value=_resolved_config())
    return coordinator


@pytest.fixture
def ai_service(stub_provider: StubProvider, resolution_coordinator: AsyncMock) -> AIService:
    registry = ProviderRegistry()
    registry.register(StubProvider)
    factory = ProviderFactory(registry)
    factory.create = MagicMock(return_value=stub_provider)
    return AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        settings=_settings(),
    )


@pytest.mark.asyncio
async def test_chat_uses_default_provider_and_model(
    ai_service: AIService,
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    result = await ai_service.chat("hello")

    resolution_coordinator.resolve.assert_awaited_once_with(
        request_provider=None,
        request_model=None,
    )
    assert result["response"] == "stub-response"
    assert result["model"] == "qwen3:8b"
    assert result["provider"] == ProviderType.OLLAMA
    assert len(stub_provider.chat_calls) == 1
    assert stub_provider.chat_calls[0].model == "qwen3:8b"


@pytest.mark.asyncio
async def test_chat_honors_explicit_provider_and_model(
    ai_service: AIService,
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    resolution_coordinator.resolve.return_value = _resolved_config(model_name="custom-model")

    result = await ai_service.chat(
        "hello",
        provider="ollama",
        model="custom-model",
    )

    resolution_coordinator.resolve.assert_awaited_with(
        request_provider="ollama",
        request_model="custom-model",
    )
    assert result["model"] == "custom-model"
    assert stub_provider.chat_calls[0].model == "custom-model"


@pytest.mark.asyncio
async def test_chat_uses_resolved_model_from_coordinator(
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    resolution_coordinator.resolve.return_value = _resolved_config(
        model_name="default-from-coordinator",
    )
    registry = ProviderRegistry()
    registry.register(StubProvider)
    factory = ProviderFactory(registry)
    factory.create = MagicMock(return_value=stub_provider)
    service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        settings=_settings(),
    )

    result = await service.chat("hello")

    assert result["model"] == "default-from-coordinator"


@pytest.mark.asyncio
async def test_check_connection_returns_legacy_payload(ai_service: AIService) -> None:
    result = await ai_service.check_connection()

    assert result["connected"] is True
    assert result["status"].value == "healthy"
    assert result["provider"] == ProviderType.OLLAMA
    assert result["model"] == "qwen3:8b"
    assert result["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_provider_unavailable_maps_to_connection_exception(
    ai_service: AIService,
    stub_provider: StubProvider,
) -> None:
    async def failing_chat(_request: ChatRequest) -> ChatResponse:
        raise ProviderUnavailableError("down", provider="ollama")

    stub_provider.chat = failing_chat  # type: ignore[method-assign]

    with pytest.raises(OllamaConnectionException):
        await ai_service.chat("hello")


@pytest.mark.asyncio
async def test_other_provider_errors_map_to_llm_response_exception(
    ai_service: AIService,
    stub_provider: StubProvider,
) -> None:
    async def failing_chat(_request: ChatRequest) -> ChatResponse:
        raise ModelNotFoundError("missing", provider="ollama")

    stub_provider.chat = failing_chat  # type: ignore[method-assign]

    with pytest.raises(LLMResponseException):
        await ai_service.chat("hello")


@pytest.mark.asyncio
async def test_provider_cleanup_on_context_manager(
    ai_service: AIService,
    stub_provider: StubProvider,
) -> None:
    await ai_service.chat("hello")

    assert stub_provider.closed is True


@pytest.mark.asyncio
async def test_provider_cleanup_without_context_manager(
    resolution_coordinator: AsyncMock,
) -> None:
    provider = ClosableOnlyProvider()
    registry = ProviderRegistry()
    registry.register(ClosableOnlyProvider)
    factory = ProviderFactory(registry)
    factory.create = MagicMock(return_value=provider)
    service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        settings=_settings(),
    )

    await service.chat("hello")

    assert provider.closed is True


@pytest.mark.asyncio
async def test_create_provider_uses_factory(
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    factory_kwargs = {"base_url": "http://custom:11434", "timeout": 30.0}
    resolution_coordinator.resolve.return_value = _resolved_config(factory_kwargs=factory_kwargs)
    registry = ProviderRegistry()
    registry.register(StubProvider)
    factory = ProviderFactory(registry)
    create_mock = MagicMock(return_value=stub_provider)
    factory.create = create_mock
    service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        settings=_settings(),
    )

    await service.chat("hello")

    create_mock.assert_called_once_with("ollama", **factory_kwargs)
