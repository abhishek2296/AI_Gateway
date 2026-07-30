"""
FastAPI application factory / entry point for the AI Gateway.

This is the single place where the ``FastAPI`` app is assembled: middleware
registration order, exception handler wiring, and router inclusion all
happen here. Per the AI Gateway architecture rule, request flow is:

    HTTP Request -> Middleware -> Route -> Service -> Provider -> APIResponse

Everything below wires up the "Middleware" and "Route" layers; the
"Service"/"Provider" layers are assembled lazily by ``api/dependencies.py``
the first time a route depends on them.
"""

from fastapi import FastAPI
from src.core.config import settings
from src.api.routes.chat import router as chat_router
from src.api.routes.models import providers_router as models_providers_router
from src.api.routes.models import router as models_router
from src.core.lifespan import lifespan
from src.schemas.common import HealthResponse
from src.core.handlers import http_exception_handler
from fastapi import HTTPException
from src.middleware.request_id import RequestIDMiddleware
from src.middleware.timing import TimingMiddleware
from src.api.routes.health import router as health_router

app = FastAPI(
    title=settings.APP_NAME,
    description="""
Backend API powering our AI Coding Assistant.

Features

- Chat with Local LLMs
- Health Monitoring
- Provider Abstraction
- Production-ready Architecture
""",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Middleware order matters here: Starlette executes middleware for the
# *incoming* request in the reverse of registration order, so registering
# RequestIDMiddleware first makes it the outermost layer. That guarantees
# request.state.request_id is set before TimingMiddleware's dispatch runs
# and logs it — reversing this order would raise an AttributeError.
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

# registering global exception handlers
app.add_exception_handler(
    HTTPException,
    http_exception_handler,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(models_providers_router)

@app.get("/")
def root():
    """
    Unauthenticated liveness/info endpoint at the API root.

    Distinct from ``GET /health`` (which probes the LLM provider): this
    endpoint only confirms the FastAPI process itself is up and reports its
    identity/version, with no downstream dependency checks — useful as a
    lightweight target for load balancer liveness probes.

    Returns:
        A plain ``dict`` (HTTP 200, not wrapped in ``APIResponse`` since this
        predates/sits outside the standard envelope) with the service's
        running status, name, and version.

    Example:
        >>> root()
        {'status': 'running', 'service': 'AI Gateway', 'version': '0.1.0'}
    """
    return {
        "status": "running",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }

# @app.get("/health")
# def health():
#     return HealthResponse(
#         status="healthy"
#     )



