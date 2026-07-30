"""
Provider abstraction layer — vendor-neutral LLM adapter contracts.

This package implements the "AI Gateway" provider abstraction described in
``05-ai-gateway-architecture.mdc``: a vendor-neutral request/response
contract (``base.py``), a normalized error hierarchy (``exceptions.py``),
shared HTTP infrastructure for cloud adapters (``http_mixin.py``,
``http_auth.py``, ``http_errors.py``, ``retry.py``, ``streaming.py``,
``validation.py``), and one concrete adapter per vendor (``ollama.py``,
``openai.py``, ``anthropic.py``, ``gemini.py``), tied together by a
name-based registry/factory (``registry.py``, ``factory.py``).

Importing the four concrete adapter modules below is not just for
re-exporting their classes — each of those modules calls
``register_provider(...)`` as a side effect of being imported (see the
bottom of e.g. ``ollama.py``). Simply importing ``src.providers`` (this
package) is therefore what populates the process-wide default registry
(``get_registry()``) with every implemented provider, which is why anything
that needs to resolve providers by name (e.g. the service layer, or
``ProviderFactory``) should import from this package rather than importing
an individual adapter module directly.
"""

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

# These four imports exist primarily for their registration side effect
# (each adapter module calls `register_provider(...)` at import time), not
# just for the class re-exports below — see the module docstring above.
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
