from enum import Enum


class ProviderType(str, Enum):
    """Runtime identifier for a known LLM backend family."""

    OLLAMA = "ollama"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
