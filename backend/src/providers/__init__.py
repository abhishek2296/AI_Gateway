"""Provider abstraction layer — vendor-neutral LLM adapter contracts."""

from src.providers.base import (
    BaseProvider,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthCheckResult,
    ModelInfo,
    TokenUsage,
)
from src.providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    StreamingNotSupportedError,
)
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry, get_registry, register_provider

__all__ = [
    "AuthenticationError",
    "BaseProvider",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "HealthCheckResult",
    "InvalidRequestError",
    "ModelInfo",
    "ModelNotFoundError",
    "ProviderError",
    "ProviderFactory",
    "ProviderNotFoundError",
    "ProviderRegistry",
    "ProviderUnavailableError",
    "RateLimitError",
    "StreamingNotSupportedError",
    "TokenUsage",
    "get_registry",
    "register_provider",
]
