from enum import Enum


class Provider(str, Enum):
    OLLAMA = "ollama"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"