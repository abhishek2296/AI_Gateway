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
    UnsupportedCapabilityError,
)
from src.providers.http_errors import HTTPErrorMapper
from src.providers.factory import ProviderFactory
from src.providers.http_mixin import HTTPProviderMixin
from src.providers.anthropic import AnthropicProvider
from src.providers.gemini import GeminiProvider
from src.providers.ollama import OllamaProvider
from src.providers.openai import OpenAIProvider
from src.providers.registry import ProviderRegistry, get_registry, register_provider

__all__ = [
    "AnthropicProvider",
    "AuthenticationError",
    "BaseProvider",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "GeminiProvider",
    "HTTPProviderMixin",
    "HealthCheckResult",
    "InvalidRequestError",
    "ModelInfo",
    "ModelNotFoundError",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderFactory",
    "ProviderNotFoundError",
    "ProviderRegistry",
    "ProviderUnavailableError",
    "RateLimitError",
    "StreamingNotSupportedError",
    "TokenUsage",
    "UnsupportedCapabilityError",
    "HTTPErrorMapper",
    "get_registry",
    "register_provider",
]
