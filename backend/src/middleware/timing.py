import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):

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