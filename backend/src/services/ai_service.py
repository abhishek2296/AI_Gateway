"""Provider-agnostic application service for AI workloads."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import src.providers.anthropic  # noqa: F401 — register AnthropicProvider
import src.providers.gemini  # noqa: F401 — register GeminiProvider
import src.providers.ollama  # noqa: F401 — register OllamaProvider
import src.providers.openai  # noqa: F401 — register OpenAIProvider

from src.core.config import Settings, get_settings
from src.core.database import AsyncSessionLocal
from src.core.enums import HealthStatus, ProviderType
from src.core.exceptions import LLMResponseException, OllamaConnectionException
from src.providers.base import BaseProvider, ChatMessage, ChatRequest
from src.providers.exceptions import ProviderError, ProviderUnavailableError
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry, get_registry
from src.services.provider_resolution_coordinator import ProviderResolutionCoordinator
from src.services.provider_resolver import ProviderDisabledError, ProviderResolutionError
from src.services.resolution_types import ResolvedConfiguration

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Resolved provider and model selection for a single service operation."""

    provider_name: str
    model_name: str
    factory_kwargs: dict[str, Any]
    request_start: float
    latency_ms: float | None = None


class AIService:
    """
    Application service that routes AI operations through :class:`BaseProvider`.

    Provider and model selection is delegated to
    :class:`~src.services.provider_resolver.ProviderResolver` via the resolution
    coordinator. Transport configuration is built by
    :class:`~src.services.provider_config_resolver.ProviderConfigResolver`.
    """

    def __init__(
        self,
        factory: ProviderFactory,
        resolution_coordinator: ProviderResolutionCoordinator,
        settings: Settings | None = None,
    ) -> None:
        self._factory = factory
        self._resolution_coordinator = resolution_coordinator
        self._settings = settings or get_settings()

    async def chat(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a chat completion and return the legacy route-compatible payload.

        Raises:
            OllamaConnectionException: When the provider backend is unavailable.
            LLMResponseException: For other provider failures.
        """
        context = await self._build_context(provider=provider, model=model)

        async def operation(provider_adapter: BaseProvider) -> dict[str, Any]:
            response = await provider_adapter.chat(
                ChatRequest(
                    model=context.model_name,
                    messages=[ChatMessage(role="user", content=message)],
                ),
            )
            return {
                "response": response.content,
                "model": response.model,
                "provider": self._to_provider_type(context.provider_name),
            }

        return await self._execute_provider_call(context, operation)

    async def check_connection(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Probe provider health and return the legacy health-route payload."""
        context = await self._build_context(provider=provider, model=model)

        async def operation(provider_adapter: BaseProvider) -> dict[str, Any]:
            result = await provider_adapter.health_check()
            return {
                "status": HealthStatus.HEALTHY if result.healthy else HealthStatus.UNHEALTHY,
                "provider": self._to_provider_type(context.provider_name),
                "model": context.model_name,
                "connected": result.healthy,
                "latency_ms": round(result.latency_ms or 0.0, 2),
            }

        return await self._execute_provider_call(context, operation)

    async def _build_context(
        self,
        *,
        provider: str | None,
        model: str | None,
    ) -> ProviderContext:
        try:
            resolved = await self._resolution_coordinator.resolve(
                request_provider=provider,
                request_model=model,
            )
        except ProviderResolutionError as exc:
            logger.error("Provider resolution failed: %s", type(exc).__name__)
            raise LLMResponseException() from exc

        logger.info(
            "Selected provider=%s, model=%s",
            resolved.provider_name,
            resolved.model_name,
        )
        return ProviderContext(
            provider_name=resolved.provider_name,
            model_name=resolved.model_name,
            factory_kwargs=resolved.factory_kwargs,
            request_start=time.perf_counter(),
        )

    def _create_provider(self, context: ProviderContext) -> BaseProvider:
        return self._factory.create(context.provider_name, **context.factory_kwargs)

    async def _execute_provider_call(
        self,
        context: ProviderContext,
        operation: Callable[[BaseProvider], Awaitable[ResultT]],
    ) -> ResultT:
        logger.info(
            "Provider request started (provider=%s, model=%s)",
            context.provider_name,
            context.model_name,
        )
        provider = self._create_provider(context)
        try:
            result = await self._run_with_provider(provider, operation)
        except ProviderError as exc:
            logger.error(
                "Provider request failed (provider=%s, model=%s, error=%s)",
                context.provider_name,
                context.model_name,
                type(exc).__name__,
            )
            raise self._map_provider_error(exc) from exc

        latency_ms = (time.perf_counter() - context.request_start) * 1000
        logger.info(
            "Provider request completed (provider=%s, model=%s, latency_ms=%.2f)",
            context.provider_name,
            context.model_name,
            latency_ms,
        )
        return result

    async def _run_with_provider(
        self,
        provider: BaseProvider,
        operation: Callable[[BaseProvider], Awaitable[ResultT]],
    ) -> ResultT:
        provider_type = type(provider)
        if (
            hasattr(provider_type, "__aenter__")
            and hasattr(provider_type, "__aexit__")
            and provider_type.__dict__.get("__aenter__") is not None
        ):
            async with provider:
                return await operation(provider)
        try:
            return await operation(provider)
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    @staticmethod
    def _map_provider_error(exc: ProviderError) -> Exception:
        if isinstance(exc, ProviderUnavailableError):
            return OllamaConnectionException()
        return LLMResponseException()

    @staticmethod
    def _to_provider_type(provider_name: str) -> ProviderType:
        return ProviderType(provider_name)


def create_ai_service(
    *,
    factory: ProviderFactory | None = None,
    registry: ProviderRegistry | None = None,
    settings: Settings | None = None,
    resolution_coordinator: ProviderResolutionCoordinator | None = None,
) -> AIService:
    """Build an :class:`AIService` with shared registry/factory defaults."""
    resolved_settings = settings or get_settings()
    resolved_registry = registry or get_registry()
    resolved_factory = factory or ProviderFactory(resolved_registry)
    coordinator = resolution_coordinator or ProviderResolutionCoordinator(
        session_factory=AsyncSessionLocal,
        settings=resolved_settings,
    )
    return AIService(
        factory=resolved_factory,
        resolution_coordinator=coordinator,
        settings=resolved_settings,
    )
