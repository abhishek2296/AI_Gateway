"""
``GET /health`` route — report connectivity to the configured LLM provider.

A thin HTTP handler per the Layer Boundaries rule: this module only maps the
HTTP response shape (via ``HealthResponse``/``APIResponse``) onto a call
into ``BaseLLMService.check_connection``. The actual provider probing logic
lives in the service layer (``services/ai_service.py``, adapted via
``services/llm_adapter.py``), not here.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_llm_service
from src.schemas.common import APIResponse
from src.schemas.health import HealthResponse
from src.services.base_llm import BaseLLMService

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    summary="Check AI Gateway health",
    description="Checks connectivity with the configured LLM provider.",
    response_model=APIResponse[HealthResponse],
)
async def health(
    llm: BaseLLMService = Depends(get_llm_service),
):
    """
    Probe the configured LLM provider and report its connectivity status.

    ``llm`` is injected via ``Depends(get_llm_service)`` as a
    ``BaseLLMService`` (see ``api/dependencies.py``), so this route depends
    only on the stable abstract contract rather than a concrete provider
    implementation — swapping the underlying provider requires no change
    here.

    Args:
        llm: Injected service used to run the connectivity probe; supplied
            automatically by FastAPI's dependency injection, never passed
            explicitly by callers.

    Returns:
        ``APIResponse[HealthResponse]`` (HTTP 200) describing whether the
        provider is reachable, which provider/model was checked, and the
        round-trip latency in milliseconds. Note this endpoint returns
        HTTP 200 even when ``connected`` is ``False`` — the *request*
        succeeded (the health check ran), even if the *provider* itself is
        down; clients should inspect ``data.status`` /
        ``data.connected`` to determine actual health.

    Raises:
        No exception is expected in the normal failure case (an
        unreachable provider is reported as ``connected: False`` in the
        response body, not as an HTTP error) — see the note above. Any
        unexpected exception from the underlying service still propagates
        as a generic ``HTTPException`` handled by
        ``core.handlers.http_exception_handler``.

    Example:
        Successful response body when the provider is reachable:

        ```json
        {
            "success": true,
            "data": {
                "status": "healthy",
                "provider": "ollama",
                "model": "qwen3:8b",
                "connected": true,
                "latency_ms": 23.41
            }
        }
        ```
    """

    result = await llm.check_connection()

    return APIResponse(
        data=HealthResponse(**result)
    )
