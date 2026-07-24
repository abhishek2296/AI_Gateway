"""Abstract provider contract for multi-vendor LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat completion request."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts returned by a provider completion."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """
    Normalized chat completion input.

    Provider adapters translate this contract into vendor-specific API payloads.
    """

    model: str
    messages: Sequence[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Normalized non-streaming chat completion output."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_response_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    """One incremental chunk from a streaming chat completion."""

    content: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingsRequest:
    """Normalized embeddings input."""

    model: str
    input: str | Sequence[str]


@dataclass(frozen=True, slots=True)
class EmbeddingsResponse:
    """Normalized embeddings output."""

    model: str
    embeddings: Sequence[Sequence[float]]
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Catalog metadata for one model offered by a provider."""

    id: str
    name: str
    display_name: str | None = None
    context_window: int | None = None
    supports_streaming: bool = False
    supports_embeddings: bool = False


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a provider connectivity and readiness probe."""

    healthy: bool
    latency_ms: float | None = None
    message: str | None = None
    models_available: int | None = None
    details: dict[str, str] = field(default_factory=dict)


class BaseProvider(ABC):
    """
    Abstract adapter for an external LLM provider.

    Each concrete implementation (Ollama, OpenAI, Anthropic, etc.) encapsulates
    vendor SDK/HTTP details and exposes a uniform async interface to the gateway
    service layer. Implementations must raise :class:`ProviderError` subclasses
    rather than leaking raw vendor exceptions.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier (e.g. ``ollama``, ``openai``)."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a non-streaming chat completion.

        Raises:
            ProviderError: On provider-side failure.
        """

    @abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """
        Execute a streaming chat completion.

        Yields incremental content chunks until the response is complete.

        Raises:
            ProviderError: On provider-side failure.
            StreamingNotSupportedError: When streaming is unavailable.
        """

    @abstractmethod
    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """
        Generate vector embeddings for the given input.

        Raises:
            ProviderError: On provider-side failure.
        """

    @abstractmethod
    async def list_models(self) -> Sequence[ModelInfo]:
        """
        Return models currently available from this provider.

        Raises:
            ProviderError: When the catalog cannot be retrieved.
        """

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Probe provider connectivity and basic readiness.

        Must not raise for expected unhealthy states; encode them in the result.
        Unexpected failures may still raise :class:`ProviderError`.
        """
