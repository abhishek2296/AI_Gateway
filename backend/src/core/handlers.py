"""
Global FastAPI exception handlers for the AI Gateway.

This module normalizes every ``HTTPException`` (including gateway-specific
subclasses such as ``OllamaConnectionException`` in ``core/exceptions.py``)
into the standard error envelope the gateway uses across all endpoints:
``{"success": False, "error": {"code": ..., "message": ...}}``. Registering
handlers here (rather than in individual routes) keeps error formatting
consistent regardless of which route or service raised the exception, per the
security rule that error handling must return generic, client-safe messages
without leaking stack traces or internals.

Handlers defined here are wired up in ``main.py`` via
``app.add_exception_handler(HTTPException, http_exception_handler)``.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Convert any raised ``HTTPException`` into the gateway's standard error envelope.

    FastAPI invokes this handler automatically whenever an ``HTTPException``
    (or a subclass, e.g. ``OllamaConnectionException``) propagates out of a
    route or dependency. It replaces FastAPI's default ``{"detail": ...}``
    body with a consistent ``{"success": False, "error": {...}}`` shape so
    every API consumer can rely on one error format across all endpoints.

    Args:
        request: The incoming request that triggered the exception. Not used
            to build the response body today, but required by FastAPI's
            exception-handler signature (``(Request, Exception) -> Response``).
        exc: The ``HTTPException`` that was raised. ``exc.status_code`` is
            echoed back as the HTTP status, and ``exc.detail`` becomes the
            human-readable error message. ``exc.__class__.__name__`` is used
            as a machine-readable error ``code`` (e.g. ``"OllamaConnectionException"``)
            so clients can branch on error type without parsing the message.

    Returns:
        A ``JSONResponse`` with the same status code as ``exc.status_code``
        and a body of the form
        ``{"success": False, "error": {"code": str, "message": str}}``.

    Example:
        Given a route that raises ``HTTPException(status_code=404, detail="Not found")``,
        the client receives:

        ```json
        {"success": false, "error": {"code": "HTTPException", "message": "Not found"}}
        ```
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.__class__.__name__,
                "message": exc.detail,
            },
        },
    )