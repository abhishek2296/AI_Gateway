"""
Response schema for the ``GET /health`` endpoint.

Defines the payload returned inside an ``APIResponse[HealthResponse]``
envelope (see ``schemas/common.py``) describing whether the gateway can
currently reach its configured LLM provider.
"""

from pydantic import BaseModel

from src.core.enums import HealthStatus, ProviderType


class HealthResponse(BaseModel):
    """
    Result of probing connectivity to the currently configured LLM provider.

    Built by ``api/routes/health.py`` from the dict returned by
    ``AIService.check_connection`` (via the ``BaseLLMService.check_connection``
    contract), then wrapped in ``APIResponse[HealthResponse]`` before being
    sent to the client.

    Attributes:
        status: Overall health verdict as a ``HealthStatus`` enum member —
            ``HealthStatus.HEALTHY`` if the provider responded successfully,
            ``HealthStatus.UNHEALTHY`` otherwise. Using the enum (rather than
            a free-text string) guarantees the client only ever sees one of
            these two literal values.
        provider: Which ``ProviderType`` was probed (e.g.
            ``ProviderType.OLLAMA``) — reflects whichever provider is
            currently configured/resolved, not necessarily a fixed constant.
        model: The model name that was used for the connectivity check, e.g.
            ``"qwen3:8b"``.
        connected: ``True`` if the provider call succeeded, ``False``
            otherwise. This is the raw boolean underlying ``status`` and is
            kept alongside it so clients can branch on connectivity without
            string/enum comparisons.
        latency_ms: Round-trip time of the health probe in milliseconds,
            rounded to 2 decimal places by the service layer. Useful for
            monitoring/alerting on provider latency degradation.

    Example:
        >>> HealthResponse(
        ...     status=HealthStatus.HEALTHY,
        ...     provider=ProviderType.OLLAMA,
        ...     model="qwen3:8b",
        ...     connected=True,
        ...     latency_ms=23.41,
        ... )
        HealthResponse(status=<HealthStatus.HEALTHY: 'healthy'>, provider=<ProviderType.OLLAMA: 'ollama'>, model='qwen3:8b', connected=True, latency_ms=23.41)
    """

    status: HealthStatus
    provider: ProviderType
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
