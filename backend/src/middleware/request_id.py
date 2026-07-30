"""
Request-correlation middleware for the AI Gateway.

Assigns a unique identifier to every incoming HTTP request so it can be
traced end-to-end across logs (see ``middleware/timing.py``, which logs this
same ID) and echoed back to the client via the ``X-Request-ID`` response
header, per the "HTTP Headers" section of the project's security rule on
traceability.

Must be registered in ``main.py`` *before* ``TimingMiddleware`` (Starlette
applies middleware in reverse of registration order for the request path, so
whichever is added first ends up running first), because ``TimingMiddleware``
reads ``request.state.request_id`` set here — without this ordering, the
timing log would raise an ``AttributeError``.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Starlette ASGI middleware that stamps every request with a unique ID.

    Subclasses ``BaseHTTPMiddleware`` (rather than raw ASGI) for simplicity,
    matching the style of ``TimingMiddleware``. Holds no instance state of
    its own — the generated ID lives on the per-request ``request.state``
    object so it is naturally request-scoped and safe under concurrent
    requests.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(RequestIDMiddleware)  # doctest: +SKIP
    """

    async def dispatch(self, request, call_next):
        """
        Generate a request ID, attach it to the request, and echo it in the response.

        Args:
            request: The incoming Starlette ``Request``. Mutated in place by
                setting ``request.state.request_id`` so downstream
                middleware (``TimingMiddleware``), dependencies, and route
                handlers can read the same ID for logging/correlation.
            call_next: Starlette-provided callable that invokes the next
                middleware/route in the chain and returns its ``Response``.

        Returns:
            The ``Response`` produced by ``call_next``, with an added
            ``X-Request-ID`` header so clients can quote this ID back when
            reporting issues.

        Raises:
            Exception: Any exception raised further down the middleware/route
                chain propagates unchanged — this middleware does not catch
                or suppress errors, it only annotates the request/response.

        Example:
            A client making a request receives a response with a header like:

            ``X-Request-ID: 3fa85f64-5717-4562-b3fc-2c963f66afa6``
        """
        # A fresh UUID per request (not reused across requests) is what makes
        # this usable for correlating a single request's log lines.
        request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return response
