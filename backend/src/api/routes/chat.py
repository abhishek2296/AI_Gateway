"""
``POST /chat`` route — send a prompt to the configured LLM.

A thin HTTP handler per the Layer Boundaries rule: this module only maps the
HTTP request/response shape (via ``ChatRequest``/``ChatResponse``/
``APIResponse``) onto a call into ``ChatService``. All provider selection,
prompt orchestration, and error mapping live in the service layer
(``services/chat_service.py`` -> ``services/ai_service.py``), not here.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_chat_service
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.chat_service import ChatService
from src.schemas.common import APIResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    summary="Chat with the configured LLM",
    description="Send a prompt to the configured language model and receive a response.",
    response_model=APIResponse[ChatResponse],
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Send a user prompt to the configured LLM and return its reply.

    FastAPI validates ``request`` against ``ChatRequest`` before this
    function runs (returning HTTP 422 automatically for malformed bodies),
    and injects ``chat_service`` via ``Depends(get_chat_service)`` — see
    ``api/dependencies.py`` for why that dependency is not cached.

    Args:
        request: The validated request body containing the user's
            ``message`` to send to the LLM.
        chat_service: Injected ``ChatService`` that orchestrates the actual
            provider call; supplied automatically by FastAPI's dependency
            injection, never passed explicitly by callers.

    Returns:
        ``APIResponse[ChatResponse]`` (HTTP 200) wrapping the LLM's reply,
        the model that generated it, and which provider served it.

    Raises:
        OllamaConnectionException: Propagates from ``ChatService`` /
            ``AIService`` as an ``HTTPException`` (503) when the configured
            provider backend is unreachable; converted to the standard error
            envelope by ``core.handlers.http_exception_handler``.
        LLMResponseException: Propagates as an ``HTTPException`` (500) for
            other provider failures (e.g. malformed provider response).

    Example:
        Request body:

        ```json
        {"message": "Explain FastAPI in one sentence."}
        ```

        Successful response body:

        ```json
        {
            "success": true,
            "data": {
                "response": "FastAPI is a modern Python framework for building APIs.",
                "model": "qwen3:8b",
                "provider": "ollama"
            }
        }
        ```
    """

    result = await chat_service.chat(
        request.message,
        model=request.model,
        provider=request.provider.value if request.provider is not None else None,
    )

    chat_response = ChatResponse(**result)

    return APIResponse(
        data=chat_response
    )
