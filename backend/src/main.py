from fastapi import FastAPI
from src.core.config import settings
from src.api.routes.chat import router as chat_router
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

#Adding middleware for UUID request ID and timing
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

# registering global exception handlers
app.add_exception_handler(
    HTTPException,
    http_exception_handler,
)

app.include_router(health_router)
app.include_router(chat_router)

@app.get("/")
def root():
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



