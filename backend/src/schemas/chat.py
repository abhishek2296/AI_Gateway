"""
Request/response schemas for the ``POST /chat`` endpoint.

``ChatRequest`` validates the inbound payload before it reaches
``api/routes/chat.py``; ``ChatResponse`` is the payload wrapped inside an
``APIResponse[ChatResponse]`` envelope (see ``schemas/common.py``) on the way
back out.
"""

from pydantic import BaseModel

from src.core.enums import ProviderType


class ChatRequest(BaseModel):
    """
    Inbound payload for ``POST /chat``.

    Pydantic validates this schema automatically before ``api/routes/chat.py``
    ever runs, so the route handler can assume ``message`` is always a
    present string field (FastAPI returns HTTP 422 for missing/invalid
    fields before the handler executes).

    Attributes:
        message: The user's prompt to send to the configured LLM. No length
            or content constraints are enforced yet (see the security rule
            on validating message length once routes mature); currently any
            non-null string, including an empty one, is accepted.

    Example:
        >>> ChatRequest(message="Explain FastAPI in one sentence.")
        ChatRequest(message='Explain FastAPI in one sentence.')
    """

    message: str
    model: str | None = None
    provider: ProviderType | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Explain FastAPI in one sentence.",
                "model": "qwen3:8b",
                "provider": "ollama",
            }
        }
    }


class ChatResponse(BaseModel):
    """
    Outbound payload for ``POST /chat``, nested inside ``APIResponse[ChatResponse]``.

    Constructed in ``api/routes/chat.py`` from the dict returned by
    ``ChatService.chat`` (itself delegating to ``AIService.chat``), which
    already contains keys matching these field names.

    Attributes:
        response: The LLM-generated reply text for the submitted
            ``ChatRequest.message``.
        model: The concrete model name that produced the response, e.g.
            ``"qwen3:8b"``. Useful when the gateway resolves a default model
            rather than the client specifying one explicitly.
        provider: Which ``ProviderType`` served the request, e.g.
            ``ProviderType.OLLAMA``.

    Example:
        >>> ChatResponse(
        ...     response="FastAPI is a modern Python framework for building APIs.",
        ...     model="qwen3:8b",
        ...     provider=ProviderType.OLLAMA,
        ... )
        ChatResponse(response='FastAPI is a modern Python framework for building APIs.', model='qwen3:8b', provider=<ProviderType.OLLAMA: 'ollama'>)
    """

    response: str
    model: str
    provider: ProviderType

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "FastAPI is a modern Python framework for building APIs.",
                "model": "qwen3:8b",
                "provider": "ollama",
            }
        }
    }
