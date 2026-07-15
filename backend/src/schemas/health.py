from pydantic import BaseModel
from src.core.enums import Provider, HealthStatus


class HealthResponse(BaseModel):

    status: HealthStatus
    provider: Provider
    model: str
    connected: bool
    latency_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "provider": "ollama",
                "model": "qwen3:8b",
                "connected": True,
                "latency_ms": 23.41
            }
        }
    }