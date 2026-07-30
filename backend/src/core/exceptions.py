"""
Domain-specific HTTP exceptions for the AI Gateway's core/service layer.

Per the "Error Handling" security rule, routes and services should raise a
custom ``HTTPException`` subclass (defined here) rather than a bare
``HTTPException`` with an inline status code and message. This keeps error
status codes and client-facing messages consistent everywhere a given
failure occurs, and keeps the messages themselves generic/safe (no stack
traces, connection strings, or other internals). These exceptions are caught
by the global handler in ``core/handlers.py`` and rendered into the
gateway's standard error envelope.
"""

from fastapi import HTTPException


class OllamaConnectionException(HTTPException):
    """
    Raised when the gateway cannot reach the configured Ollama server.

    Maps to HTTP ``503 Service Unavailable`` — the failure is on the
    upstream provider side (Ollama not running, wrong host/port, network
    issue), not a problem with the client's request, so ``503`` is more
    accurate than a ``4xx`` client error or a generic ``500``.

    Example:
        Raised from ``OllamaService`` when a connection attempt fails:

        >>> raise OllamaConnectionException()  # doctest: +SKIP
        # -> HTTP 503, body includes:
        #    "Unable to connect to Ollama. Please ensure Ollama is running."
    """

    def __init__(self) -> None:
        """
        Build the exception with a fixed status code and client-safe message.

        Takes no arguments because the failure mode is always the same (an
        unreachable Ollama server) and the message intentionally avoids
        including the underlying connection error's raw text — the raw error
        may contain internal details (host, port, stack trace) that should
        not be exposed to API clients per the security rules; instead, the
        full exception should be logged server-side with
        ``logger.exception()`` at the call site before raising this.
        """
        super().__init__(
            status_code=503,
            detail="Unable to connect to Ollama. Please ensure Ollama is running."
        )


class LLMResponseException(HTTPException):
    """
    Raised when a provider is reachable but fails to produce a valid chat response.

    Maps to HTTP ``500 Internal Server Error`` since, from the client's
    perspective, this is an unexpected failure rather than something they can
    fix by changing their request (compare with ``OllamaConnectionException``,
    which specifically signals an upstream availability problem).

    Example:
        Raised from a provider service when the LLM call itself errors out
        (e.g. malformed response, unexpected exception during generation):

        >>> raise LLMResponseException()  # doctest: +SKIP
        # -> HTTP 500, body includes:
        #    "Failed to generate response from the language model."
    """

    def __init__(self) -> None:
        """
        Build the exception with a fixed status code and client-safe message.

        As with ``OllamaConnectionException``, the underlying error detail is
        intentionally omitted from the client-facing message; callers should
        log the original exception server-side before raising this one.
        """
        super().__init__(
            status_code=500,
            detail="Failed to generate response from the language model."
        )


class ModelNotFoundHTTPException(HTTPException):
    """Raised when a requested catalog model does not exist (HTTP 404)."""

    def __init__(self, detail: str = "The requested model was not found in the catalog.") -> None:
        super().__init__(status_code=404, detail=detail)


class ModelDisabledHTTPException(HTTPException):
    """Raised when a catalog model exists but is disabled (HTTP 400)."""

    def __init__(self, detail: str = "The requested model is disabled.") -> None:
        super().__init__(status_code=400, detail=detail)


class AmbiguousModelHTTPException(HTTPException):
    """Raised when a model name matches multiple providers (HTTP 409)."""

    def __init__(
        self,
        detail: str = "Model name is ambiguous; specify the provider query parameter.",
    ) -> None:
        super().__init__(status_code=409, detail=detail)