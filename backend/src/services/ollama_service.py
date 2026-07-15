import logging
import time
import httpx
from ollama import AsyncClient, ResponseError

from src.core.config import settings
from src.core.exceptions import (
    LLMResponseException,
    OllamaConnectionException,
)
from src.services.base_llm import BaseLLMService
from src.core.enums import Provider, HealthStatus

logger = logging.getLogger(__name__)


class OllamaService(BaseLLMService):
    def __init__(self):
        self.client = AsyncClient(
            host=settings.OLLAMA_HOST,
            timeout=settings.TIMEOUT,
        )
        self.model = settings.OLLAMA_MODEL


    async def chat(self, message: str) -> dict:
        try:
            logger.info("Sending request to model '%s'", self.model)

            response = await self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
            )

            logger.info("Response received successfully")

        except httpx.ConnectError as e:
            logger.exception("Failed to connect to Ollama: %s", e)
            raise OllamaConnectionException()

        except ResponseError as e:
            logger.exception("Ollama returned an error: %s", e)
            raise LLMResponseException()

        return {
            "response": response["message"]["content"],
            "model": self.model,
            "provider": Provider.OLLAMA,
        }


    async def check_connection(self) -> dict:

        start = time.perf_counter()

        try:

            await self.client.ps()

            latency = (time.perf_counter() - start) * 1000

            return {
                "status": HealthStatus.HEALTHY,
                "provider": Provider.OLLAMA,
                "model": self.model,
                "connected": True,
                "latency_ms": round(latency, 2),
            }

        except (httpx.ConnectError, ResponseError):

            latency = (time.perf_counter() - start) * 1000

            return {
                "status": HealthStatus.UNHEALTHY,
                "provider": Provider.OLLAMA,
                "model": self.model,
                "connected": False,
                "latency_ms": round(latency, 2),
            }
