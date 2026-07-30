"""
Mock provider used as a lightweight test double for registry/factory tests.

``ProviderRegistry`` and ``ProviderFactory`` (see
``backend/tests/unit/providers/test_provider_registry.py``) only need *some*
concrete ``BaseProvider`` subclass to register and instantiate — they are not
supposed to depend on any real vendor SDK or network call. ``MockProvider``
fulfils the ``BaseProvider`` abstract interface with trivial, deterministic
implementations so those tests can exercise registration/lookup/instantiation
logic in complete isolation from any real provider (Ollama/OpenAI/etc.).
"""

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
    """
    Minimal concrete ``BaseProvider`` implementation for registry/factory tests.

    Every abstract method returns a fixed, hard-coded value instead of making
    any real HTTP call. This keeps registry/factory tests fast, deterministic,
    and independent of any vendor API — they only need to confirm that a
    provider *class* can be registered and instantiated correctly, not that a
    specific vendor integration works.

    Attributes:
        provider_name: Registry key (``"mock"``) used by tests to register,
            look up, and create instances of this provider.
        endpoint: Arbitrary constructor argument used by factory tests to
            confirm that keyword arguments passed to
            ``ProviderFactory.create()`` are forwarded to the provider
            constructor.

    Example:
        >>> provider = MockProvider(endpoint="http://test")
        >>> provider.endpoint
        'http://test'
    """

    provider_name = "mock"

    def __init__(self, *, endpoint: str = "http://mock") -> None:
        """
        Store the endpoint so factory tests can verify kwarg forwarding.

        Args:
            endpoint: Arbitrary string recorded on the instance; never used
                to make a real network call.
        """
        self.endpoint = endpoint

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Return a fixed ``ChatResponse`` echoing the requested model name."""
        return ChatResponse(content="mock", model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Yield a single fixed chunk to satisfy the streaming interface."""
        yield ChatStreamChunk(content="mock")

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Return a fixed single-vector embedding response."""
        return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])

    async def list_models(self) -> Sequence[ModelInfo]:
        """Return a single fixed ``ModelInfo`` entry."""
        return (ModelInfo(id="mock-model", name="mock-model"),)

    async def health_check(self) -> HealthCheckResult:
        """Report a fixed healthy status."""
        return HealthCheckResult(healthy=True, latency_ms=1.0)
