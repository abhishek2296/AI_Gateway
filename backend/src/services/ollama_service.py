"""
Concrete, standalone Ollama implementation of :class:`~src.services.base_llm.BaseLLMService`.

This is the gateway's original, pre-multi-provider Ollama integration: it
talks directly to a single Ollama instance configured entirely from static
application settings (``settings.OLLAMA_HOST``, ``settings.OLLAMA_MODEL``),
with no per-request provider/model resolution and no dependency on the
:class:`~src.providers.base.BaseProvider` abstraction used by
:class:`~src.services.ai_service.AIService`.

It is kept as a simple, self-contained reference implementation of the
``BaseLLMService`` contract (see ``05-ai-gateway-architecture.mdc`` for how
this fits alongside the newer ``BaseProvider``-based ``OllamaProvider`` used
by the multi-provider path).
"""

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
from src.core.enums import HealthStatus, ProviderType

logger = logging.getLogger(__name__)


class OllamaService(BaseLLMService):
    """
    Direct, single-instance Ollama chat and health-check client.

    Unlike :class:`~src.services.ai_service.AIService`, this class hardcodes
    its provider identity (always ``ProviderType.OLLAMA``) and reads its
    connection settings once, at construction time, from the global
    ``settings`` object — there is no per-call provider/model override and no
    database-backed configuration.

    Attributes:
        client: The underlying ``ollama.AsyncClient`` used to talk to the
            Ollama HTTP API, configured with the host and timeout from
            application settings.
        model: The Ollama model name (e.g. ``"qwen3:8b"``) used for every
            ``chat`` call, taken from ``settings.OLLAMA_MODEL``.

    Example:
        >>> service = OllamaService()
        >>> result = await service.chat("Hello!")
        >>> result["provider"]
        <ProviderType.OLLAMA: 'ollama'>
    """

    def __init__(self):
        """
        Configure the Ollama client from global application settings.

        Reads ``settings.OLLAMA_HOST``, ``settings.TIMEOUT``, and
        ``settings.OLLAMA_MODEL`` once at construction time; there is no
        mechanism to change these afterwards without creating a new
        ``OllamaService`` instance.
        """
        self.client = AsyncClient(
            host=settings.OLLAMA_HOST,
            timeout=settings.TIMEOUT,
        )
        self.model = settings.OLLAMA_MODEL


    async def chat(self, message: str) -> dict:
        """
        Send a single-turn chat message to Ollama and return the normalized result.

        Any transport-level connection failure or Ollama-reported API error
        is caught and remapped to one of the application's normalized
        exception types, so callers never need to import or catch the
        underlying ``httpx``/``ollama`` SDK exception types directly.

        Args:
            message: The user's plain-text prompt, sent as a single
                ``user``-role message (no multi-turn history is supported).

        Returns:
            A ``dict`` with:
                - ``response`` (str): The model's reply text.
                - ``model`` (str): The configured model name (``self.model``).
                - ``provider`` (:class:`~src.core.enums.ProviderType`):
                  Always ``ProviderType.OLLAMA``.

        Raises:
            OllamaConnectionException: If the Ollama host cannot be reached
                (``httpx.ConnectError``) — e.g. the service is not running.
            LLMResponseException: If Ollama reaches the request but returns
                an application-level error (``ollama.ResponseError``) — e.g.
                the configured model is not pulled/available.

        Example:
            >>> await service.chat("What is 2+2?")
            {'response': '4', 'model': 'qwen3:8b', 'provider': <ProviderType.OLLAMA: 'ollama'>}
        """
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
            # A connection-level failure (host down, wrong port, etc.) is
            # distinct from an application-level Ollama error below -- callers
            # may want to react differently (e.g. retry/backoff) to "backend
            # unreachable" versus "backend returned an error".
            logger.exception("Failed to connect to Ollama: %s", e)
            raise OllamaConnectionException()

        except ResponseError as e:
            logger.exception("Ollama returned an error: %s", e)
            raise LLMResponseException()

        return {
            "response": response["message"]["content"],
            "model": self.model,
            "provider": ProviderType.OLLAMA,
        }


    async def check_connection(self) -> dict:
        """
        Probe Ollama connectivity by listing running models (``client.ps()``).

        Unlike :meth:`chat`, this method never raises for expected failure
        modes — connection and response errors are caught and folded into an
        ``UNHEALTHY``/``connected=False`` result instead, since health checks
        are expected to report status rather than propagate exceptions.

        Returns:
            A ``dict`` with:
                - ``status`` (:class:`~src.core.enums.HealthStatus`):
                  ``HEALTHY`` if the probe succeeded, else ``UNHEALTHY``.
                - ``provider`` (:class:`~src.core.enums.ProviderType`):
                  Always ``ProviderType.OLLAMA``.
                - ``model`` (str): The configured model name.
                - ``connected`` (bool): Whether the probe succeeded.
                - ``latency_ms`` (float): Round-trip time in milliseconds,
                  rounded to 2 decimal places, measured regardless of
                  success or failure (so a slow *failure* is still visible in
                  the latency figure).

        Example:
            >>> await service.check_connection()
            {'status': <HealthStatus.HEALTHY: 'healthy'>, 'provider': <ProviderType.OLLAMA: 'ollama'>, 'model': 'qwen3:8b', 'connected': True, 'latency_ms': 5.12}
        """

        start = time.perf_counter()

        try:

            await self.client.ps()

            latency = (time.perf_counter() - start) * 1000

            return {
                "status": HealthStatus.HEALTHY,
                "provider": ProviderType.OLLAMA,
                "model": self.model,
                "connected": True,
                "latency_ms": round(latency, 2),
            }

        except (httpx.ConnectError, ResponseError):

            latency = (time.perf_counter() - start) * 1000

            return {
                "status": HealthStatus.UNHEALTHY,
                "provider": ProviderType.OLLAMA,
                "model": self.model,
                "connected": False,
                "latency_ms": round(latency, 2),
            }
