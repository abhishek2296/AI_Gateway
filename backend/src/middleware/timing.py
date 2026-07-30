"""
Response-timing middleware for the AI Gateway.

Measures how long each request takes to process, logs it alongside the
request's correlation ID (see ``middleware/request_id.py``), and exposes the
duration to the client via an ``X-Response-Time`` header. This gives basic,
zero-configuration observability for every endpoint without instrumenting
individual routes.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Starlette ASGI middleware that measures and logs per-request latency.

    Must be registered in ``main.py`` *after* ``RequestIDMiddleware`` (i.e.
    ``app.add_middleware(RequestIDMiddleware)`` then
    ``app.add_middleware(TimingMiddleware)``), because Starlette runs
    middleware for the incoming request in the *reverse* of registration
    order — registering ``RequestIDMiddleware`` first makes it the
    outermost layer, so it sets ``request.state.request_id`` before this
    middleware's ``dispatch`` runs and tries to read it.

    Example:
        >>> from fastapi import FastAPI
        >>> from src.middleware.request_id import RequestIDMiddleware
        >>> app = FastAPI()
        >>> app.add_middleware(RequestIDMiddleware)  # doctest: +SKIP
        >>> app.add_middleware(TimingMiddleware)      # doctest: +SKIP
    """

    async def dispatch(self, request, call_next):
        """
        Time the downstream request handling and log/report the elapsed duration.

        Args:
            request: The incoming Starlette ``Request``. Expected to already
                have ``request.state.request_id`` set by
                ``RequestIDMiddleware`` — this is why middleware
                registration order matters (see class docstring).
            call_next: Starlette-provided callable that invokes the next
                middleware/route in the chain and returns its ``Response``.

        Returns:
            The ``Response`` produced by ``call_next``, with an added
            ``X-Response-Time`` header (formatted as seconds with 3 decimal
            places, e.g. ``"0.042s"``).

        Raises:
            AttributeError: If ``request.state.request_id`` was never set
                (i.e. ``RequestIDMiddleware`` is missing or registered in
                the wrong order), the logging call below will fail.
            Exception: Any exception raised by the downstream handler
                propagates unchanged after timing — this middleware does not
                swallow errors.

        Example:
            Produces a log line such as::

                [3fa85f64-...] POST /chat completed in 0.842 sec

            and adds a response header like ``X-Response-Time: 0.842s``.
        """
        start = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - start

        logger.info(
            "[%s] %s %s completed in %.3f sec",
            request.state.request_id,
            request.method,
            request.url.path,
            elapsed,
        )

        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"

        return response
