"""
Application startup/shutdown lifecycle for the AI Gateway's FastAPI app.

FastAPI's `lifespan` protocol lets an application run setup code once before
it starts accepting requests, and teardown code once after it stops. This
module wires that protocol to the gateway's infrastructure concerns:
configuring logging, performing a best-effort connectivity check against the
default LLM provider, and releasing database connections cleanly on
shutdown. It is registered in `main.py` via `FastAPI(lifespan=lifespan)`.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from src.core.logging import setup_logging
from src.core.database import close_database_connection
from src.api.dependencies import set_ai_service_instance
from src.providers.factory import ProviderFactory
from src.providers.registry import get_registry
from src.registry.memory import MemoryModelRegistry
from src.registry.read_only import ReadOnlyModelRegistryView
from src.services.ai_service import create_ai_service
from src.services.model_catalog_loader import create_model_catalog_loader
from src.services.model_registry_service import ModelRegistryService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run startup logic before the app serves requests, then shutdown logic after.

    This is an async context manager consumed directly by FastAPI: everything
    before ``yield`` runs once at process startup (before the server accepts
    any requests), and everything after ``yield`` runs once at shutdown
    (after the server stops accepting new requests). The order of operations
    matters:

    1. Logging is configured first (``setup_logging()``) so every subsequent
       startup step — including the provider connectivity check — is
       actually captured in the logs.
    2. A default ``AIService`` is built and its connection checked so
       operators immediately see, in the startup logs, whether the
       configured default provider (e.g. Ollama) is reachable. This check is
       best-effort and non-fatal: a failed connection only logs a warning
       rather than raising, so the gateway still starts and can serve
       requests for other providers/config combinations even if the default
       one is temporarily down.
    3. On shutdown, the database engine's connection pool is disposed via
       ``close_database_connection()`` so pooled connections are released
       cleanly instead of being abruptly dropped when the process exits.

    Args:
        app: The FastAPI application instance being started. Required by
            FastAPI's lifespan protocol signature, though this implementation
            does not need to read or mutate ``app`` directly.

    Yields:
        None. Control returns to FastAPI while the app serves requests;
        execution resumes here (after the ``yield``) when the app is
        shutting down.

    Raises:
        Exception: Any unexpected error raised while constructing the
            default ``AIService`` (``create_ai_service()``) — for example, a
            misconfigured default provider that cannot even be instantiated
            — propagates and prevents the app from starting. Connectivity
            failures to an otherwise-valid provider, in contrast, are caught
            internally by ``check_connection()`` and only logged.

    Example:
        Registered once, at app construction time, in ``main.py``:

        >>> from fastapi import FastAPI
        >>> app = FastAPI(lifespan=lifespan)  # doctest: +SKIP
    """

    setup_logging()

    logger.info("=" * 60)
    logger.info("Starting AI Gateway...")

    model_registry = MemoryModelRegistry()
    catalog_loader = create_model_catalog_loader(
        model_registry,
        factory=ProviderFactory(get_registry()),
    )
    await catalog_loader.load()
    readonly_registry = ReadOnlyModelRegistryView(model_registry)
    model_registry_service = ModelRegistryService(readonly_registry)
    app.state.model_registry = model_registry
    app.state.readonly_model_registry = readonly_registry
    app.state.model_registry_service = model_registry_service

    ai_service = create_ai_service(
        factory=ProviderFactory(get_registry()),
        model_registry_service=model_registry_service,
    )
    set_ai_service_instance(ai_service)
    health = await ai_service.check_connection()

    if health["connected"]:
        logger.info(
            "Connected to %s (%s) in %.2f ms",
            health["provider"],
            health["model"],
            health["latency_ms"],
        )
    else:
        logger.warning("Unable to connect to default provider.")

    logger.info("AI Gateway Ready")
    logger.info("=" * 60)

    yield

    await close_database_connection()

    logger.info("=" * 60)
    logger.info("Stopping AI Gateway...")
    logger.info("=" * 60)
