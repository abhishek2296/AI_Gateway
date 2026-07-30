"""
Shared, provider-agnostic response envelopes used across every API route.

Every endpoint in the gateway returns one of two top-level shapes so clients
never have to guess the response structure:

- ``APIResponse[T]``  — success envelope wrapping a typed ``data`` payload.
- ``FailureResponse``  — error envelope wrapping an ``ErrorResponse`` detail.

Routes declare ``response_model=APIResponse[SomeSchema]`` (see
``api/routes/chat.py`` and ``api/routes/health.py``); the generic parameter
``T`` is filled in per-route so OpenAPI docs show the exact payload shape for
that endpoint while still sharing one envelope implementation. The error
shape is produced centrally by ``core/handlers.http_exception_handler``
rather than by individual routes, so its schema lives here for reference and
potential reuse (e.g. documenting error responses in OpenAPI).
"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """
    Machine-readable error detail nested inside a :class:`FailureResponse`.

    This mirrors the ``error`` object built by
    ``core.handlers.http_exception_handler`` from a raised ``HTTPException``:
    ``code`` comes from the exception's class name and ``message`` from its
    ``detail`` string.

    Attributes:
        code: A machine-readable error identifier, typically the raising
            exception's class name (e.g. ``"OllamaConnectionException"``).
            Clients can branch on this without parsing the human-readable
            ``message``.
        message: A human-readable description of what went wrong, safe to
            display to an end user (no stack traces or internal details,
            per the project's security rule on generic error messages).

    Example:
        >>> ErrorResponse(code="OllamaConnectionException", message="Unable to connect to Ollama.")
        ErrorResponse(code='OllamaConnectionException', message='Unable to connect to Ollama.')
    """

    code: str
    message: str


class APIResponse(BaseModel, Generic[T]):
    """
    Standard success envelope returned by every gateway endpoint.

    ``APIResponse`` is generic over ``T`` so each route can specialize it
    with its own response schema (e.g. ``APIResponse[ChatResponse]``,
    ``APIResponse[HealthResponse]``) while every client only needs to learn
    one top-level shape: check ``success``, then read ``data``.

    Attributes:
        success: Whether the request succeeded. Defaults to ``True`` because
            this envelope is only ever constructed on the success path —
            failures are represented by :class:`FailureResponse` instead,
            produced by the global exception handler.
        data: The endpoint-specific payload, typed as ``T`` at the route
            level (e.g. a ``ChatResponse`` or ``HealthResponse`` instance).

    Example:
        >>> from src.schemas.chat import ChatResponse
        >>> from src.core.enums import ProviderType
        >>> envelope = APIResponse[ChatResponse](
        ...     data=ChatResponse(
        ...         response="Hi there!",
        ...         model="qwen3:8b",
        ...         provider=ProviderType.OLLAMA,
        ...     )
        ... )
        >>> envelope.success
        True
    """

    success: bool = True
    data: T


class FailureResponse(BaseModel):
    """
    Standard error envelope returned when a request fails.

    Built centrally by ``core.handlers.http_exception_handler`` for every
    ``HTTPException`` (including gateway-specific subclasses such as
    ``OllamaConnectionException``) so all failure responses share one shape
    across the entire API, regardless of which route or service raised the
    error.

    Attributes:
        success: Always ``False`` for this envelope — distinguishes it from
            :class:`APIResponse` at a glance, even without inspecting the
            HTTP status code.
        error: The structured error detail (code + message); see
            :class:`ErrorResponse`.

    Example:
        >>> FailureResponse(
        ...     error=ErrorResponse(code="LLMResponseException", message="Failed to generate response.")
        ... )
        FailureResponse(success=False, error=ErrorResponse(code='LLMResponseException', message='Failed to generate response.'))
    """

    success: bool = False
    error: ErrorResponse


class HealthResponse(BaseModel):
    """
    Minimal health-status schema.

    Note:
        This is a legacy/unused stub — the health route (``api/routes/health.py``)
        actually returns the richer ``schemas.health.HealthResponse`` (which
        also reports provider, model, connectivity, and latency). This class
        is kept only because ``main.py`` still imports it; it is not wired
        into any active route. Prefer ``schemas.health.HealthResponse`` for
        new code.

    Attributes:
        status: Free-form status string (e.g. ``"healthy"``). Unlike
            ``schemas.health.HealthResponse.status``, this is a plain ``str``
            rather than the ``HealthStatus`` enum, so it performs no
            validation of the allowed values.

    Example:
        >>> HealthResponse(status="healthy")
        HealthResponse(status='healthy')
    """

    status: str
