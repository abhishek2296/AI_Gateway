from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from src.core.logging import setup_logging
from src.core.database import close_database_connection
from src.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    setup_logging()

    logger.info("=" * 60)
    logger.info("Starting AI Gateway...")

    llm = OllamaService()

    health = await llm.check_connection()

    if health["connected"]:
        logger.info(
            "Connected to %s (%s) in %.2f ms",
            health["provider"],
            health["model"],
            health["latency_ms"],
        )
    else:
        logger.warning("Unable to connect to Ollama.")

    logger.info("AI Gateway Ready")
    logger.info("=" * 60)

    yield

    await close_database_connection()

    logger.info("=" * 60)
    logger.info("Stopping AI Gateway...")
    logger.info("=" * 60)
