"""
Shared, cross-layer enumerations for the AI Gateway's core infrastructure.

These enums are intentionally lightweight (no dependencies on services,
schemas, or the registry) so they can be imported anywhere — routes,
schemas, services, and the registry layer — without creating import cycles.
They describe two orthogonal, stable vocabularies used across the codebase:
which provider backend a request targets (``ProviderType``) and whether a
health check succeeded (``HealthStatus``).
"""

from enum import Enum


class ProviderType(str, Enum):
    """
    Single source of truth for a known LLM backend family.

    Subclassing ``str`` means members compare equal to their plain string
    value and serialize directly to JSON (e.g. in ``/health``, ``/chat``, or
    ``/models`` responses) without any custom encoder. Every layer — registry,
    providers, services, schemas, and API routes — imports this enum from
    ``src.core.enums`` so provider identity never drifts between modules.

    Members:
        OLLAMA: Local/self-hosted models served via Ollama.
        OPENAI: Models served by OpenAI's API.
        ANTHROPIC: Models served by Anthropic's API.
        GEMINI: Models served by Google's Gemini API.

    Example:
        >>> ProviderType.OLLAMA == "ollama"
        True
        >>> ProviderType("openai")
        <ProviderType.OPENAI: 'openai'>
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class HealthStatus(str, Enum):
    """
    Overall connectivity status reported by a provider health check.

    Used by health-check responses (e.g. ``check_connection()`` on
    ``BaseLLMService`` implementations and the ``/health`` route's schema) to
    communicate, in a serializable and type-safe way, whether the configured
    provider is currently reachable.

    Members:
        HEALTHY: The provider responded successfully within the configured
            timeout.
        UNHEALTHY: The provider could not be reached, timed out, or returned
            an error.

    Example:
        >>> HealthStatus.HEALTHY == "healthy"
        True
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
