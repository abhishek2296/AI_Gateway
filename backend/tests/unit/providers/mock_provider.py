"""Mock provider for registry and factory unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

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


class MockProvider(BaseProvider):
    """Minimal concrete provider used to exercise registry and factory behavior."""

    provider_name = "mock"

    def __init__(self, *, endpoint: str = "http://mock") -> None:
        self.endpoint = endpoint

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="mock", model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="mock")

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])

    async def list_models(self) -> Sequence[ModelInfo]:
        return (ModelInfo(id="mock-model", name="mock-model"),)

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, latency_ms=1.0)
