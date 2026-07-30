"""
Provider-agnostic application service for AI workloads.

``AIService`` is the top-level orchestrator in the gateway's request flow
(Route -> Service -> Provider): it is the single place where "resolve a
provider/model" and "call that provider" are joined together. Route handlers
(via :class:`~src.services.chat_service.ChatService` for chat, or health
routes directly) call into this service; it never talks to FastAPI request
objects and knows nothing about HTTP.

For each operation (``chat``, ``check_connection``) it:

1. Resolves provider/model identity and connection kwargs via
   :class:`~src.services.provider_resolution_coordinator.ProviderResolutionCoordinator`
   (which itself delegates to
   :class:`~src.services.provider_resolver.ProviderResolver` and
   :class:`~src.services.provider_config_resolver.ProviderConfigResolver`,
   with caching).
2. Instantiates the concrete :class:`~src.providers.base.BaseProvider`
   adapter via :class:`~src.providers.factory.ProviderFactory`.
3. Invokes the adapter, measuring latency and normalizing both successful
   results and provider errors into the shapes existing routes/tests expect.

The provider-specific ``import src.providers.<name>`` statements below have a
side effect only — each module registers its provider class with the global
:class:`~src.providers.registry.ProviderRegistry` at import time, which is why
they are imported here (once, at module load) even though nothing in this
file references their contents directly.
"""

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
from src.services.model_registry_service import ModelRegistryService
from src.services.provider_resolution_coordinator import ProviderResolutionCoordinator
from src.services.provider_resolver import ProviderDisabledError, ProviderResolutionError
from src.registry.exceptions import (
    AmbiguousModelError,
    ModelDisabledError,
    ModelNotFoundError,
)
from src.services.resolution_types import ResolvedConfiguration

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """
    Resolved provider and model selection for a single service operation.

    Built once per call to :meth:`AIService.chat` or
    :meth:`AIService.check_connection` (via ``_build_context``), then threaded
    through provider instantiation, invocation, and error/latency handling
    (``_create_provider``, ``_execute_provider_call``). Bundling these values
    together avoids passing four-plus separate parameters between the
    private helper methods.

    Attributes:
        provider_name: The resolved provider's stable machine key (e.g.
            ``"ollama"``), used to look up the adapter class in the
            :class:`~src.providers.factory.ProviderFactory` and in log
            messages.
        model_name: The resolved model identifier to send to the provider
            (e.g. ``"qwen3:8b"``).
        factory_kwargs: Provider-specific constructor kwargs (base URL,
            timeout, API key, etc.) built by
            :class:`~src.services.provider_config_resolver.ProviderConfigResolver`,
            ready to unpack into ``ProviderFactory.create``.
        request_start: A ``time.perf_counter()`` timestamp captured when
            resolution completed, used to compute request latency in
            ``_execute_provider_call``. Uses ``perf_counter`` rather than
            wall-clock time since only the elapsed duration matters, not the
            absolute time.
        latency_ms: Reserved for a future latency measurement stored back
            onto the context. Currently always ``None`` — latency is instead
            computed locally and logged directly in
            ``_execute_provider_call`` without being written back here.

    Example:
        >>> context = ProviderContext(
        ...     provider_name="ollama",
        ...     model_name="qwen3:8b",
        ...     factory_kwargs={"base_url": "http://localhost:11434", "timeout": 30.0},
        ...     request_start=time.perf_counter(),
        ... )
    """

    provider_name: str
    model_name: str
    factory_kwargs: dict[str, Any]
    request_start: float
    latency_ms: float | None = None


class AIService:
    """
    Application service that routes AI operations through :class:`BaseProvider`.

    This is the orchestration hub described in the module docstring: it owns
    no resolution logic and no provider SDK details itself, delegating both
    to collaborators so it stays a thin, testable coordination layer. Provider
    and model selection is delegated to
    :class:`~src.services.provider_resolver.ProviderResolver` via the
    resolution coordinator (which adds caching); transport configuration
    (base URL, timeout, credentials) is built by
    :class:`~src.services.provider_config_resolver.ProviderConfigResolver`.

    Prefer constructing instances via :func:`create_ai_service`, which wires
    up the shared registry/factory/coordinator defaults; the constructor
    itself is kept fully explicit (no internal defaulting beyond
    ``settings``) so tests can inject fakes for every collaborator.

    Example:
        >>> ai_service = create_ai_service()
        >>> result = await ai_service.chat("Hello!")
        >>> result["provider"]
        <ProviderType.OLLAMA: 'ollama'>
    """

    def __init__(
        self,
        factory: ProviderFactory,
        resolution_coordinator: ProviderResolutionCoordinator,
        model_registry_service: ModelRegistryService,
        settings: Settings | None = None,
    ) -> None:
        """
        Wire up the provider factory, resolution coordinator, registry service, and settings.

        Args:
            factory: Creates concrete :class:`~src.providers.base.BaseProvider`
                adapter instances from a provider name and constructor
                kwargs.
            resolution_coordinator: Builds provider connection kwargs (with
                caching) after registry model selection.
            model_registry_service: Resolves and validates catalog entries for
                chat and health operations (registry-only model selection).
            settings: Application settings. Defaults to
                :func:`~src.core.config.get_settings` when omitted.
        """
        self._factory = factory
        self._resolution_coordinator = resolution_coordinator
        self._model_registry_service = model_registry_service
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

        The returned ``dict`` shape (``response``/``model``/``provider`` keys)
        matches what pre-multi-provider routes and tests already expect, so
        introducing provider-agnostic routing did not require changing the
        HTTP-facing response contract.

        Args:
            message: The user's chat message text. Wrapped as a single
                ``user``-role :class:`~src.providers.base.ChatMessage` before
                being sent to the provider — this method does not support
                multi-turn history or system prompts.
            provider: Optional provider name override (e.g. ``"openai"``).
                When given, it takes top precedence in resolution (see
                :class:`~src.services.provider_resolver.ProviderResolver`).
                ``None`` defers to the database/settings/fallback tiers.
            model: Optional model name override, resolved independently of
                ``provider`` with the same precedence rules.

        Returns:
            A ``dict`` with:
                - ``response`` (str): The provider's reply text.
                - ``model`` (str): The model that actually produced the
                  response (as reported by the provider).
                - ``provider`` (:class:`~src.core.enums.ProviderType`): The
                  resolved provider identity.

        Raises:
            OllamaConnectionException: When the provider backend is
                unreachable (mapped from
                :class:`~src.providers.exceptions.ProviderUnavailableError`).
            LLMResponseException: For provider resolution failures or any
                other provider-side error.

        Example:
            >>> await ai_service.chat("Hello!", provider="ollama")
            {'response': 'Hi there!', 'model': 'qwen3:8b', 'provider': <ProviderType.OLLAMA: 'ollama'>}
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
        """
        Probe provider health and return the legacy health-route payload.

        Resolves a provider/model exactly like :meth:`chat` (same precedence
        rules), then delegates to the adapter's
        :meth:`~src.providers.base.BaseProvider.health_check`, which is
        expected to report unhealthy states in its result rather than
        raising — so a genuinely down provider still produces a normal
        ``connected=False`` payload instead of propagating an exception here.

        Args:
            provider: Optional provider name override, same precedence rules
                as :meth:`chat`.
            model: Optional model name override, same precedence rules as
                :meth:`chat`. Included in the response even though it does
                not affect the health probe itself, so callers can see which
                model would be used for a subsequent chat call.

        Returns:
            A ``dict`` with:
                - ``status`` (:class:`~src.core.enums.HealthStatus`):
                  ``HEALTHY`` or ``UNHEALTHY``.
                - ``provider`` (:class:`~src.core.enums.ProviderType`): The
                  resolved provider identity.
                - ``model`` (str): The resolved model name.
                - ``connected`` (bool): Whether the health probe succeeded.
                - ``latency_ms`` (float): Probe round-trip time, rounded to 2
                  decimal places; ``0.0`` if the adapter did not report one.

        Raises:
            LLMResponseException: If provider/model resolution itself fails
                (before the health probe is even attempted).

        Example:
            >>> await ai_service.check_connection()
            {'status': <HealthStatus.HEALTHY: 'healthy'>, 'provider': <ProviderType.OLLAMA: 'ollama'>, 'model': 'qwen3:8b', 'connected': True, 'latency_ms': 12.34}
        """
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
        """
        Resolve provider/model identity and connection kwargs into a :class:`ProviderContext`.

        Shared by both :meth:`chat` and :meth:`check_connection` so resolution
        and its error handling are implemented exactly once. Resolution
        failures are caught here and remapped from the resolver's
        :class:`~src.services.provider_resolver.ProviderResolutionError` to
        :class:`~src.core.exceptions.LLMResponseException` — the application's
        normalized error type — so callers of ``AIService`` never need to
        know about resolver-internal exception types.

        Args:
            provider: Optional provider name override from the caller, or
                ``None`` to use the database/settings/fallback precedence
                tiers.
            model: Optional model name override from the caller, or ``None``
                to use the database/settings precedence tiers.

        Returns:
            A :class:`ProviderContext` with the resolved provider/model names,
            factory kwargs, and a freshly captured ``request_start`` timestamp
            for later latency measurement.

        Raises:
            LLMResponseException: If resolution fails for any reason (no
                provider/model could be determined, or the requested
                provider is disabled).
        """
        try:
            catalog_model = await self._model_registry_service.resolve_for_chat(
                model=model,
                provider=provider,
            )
        except (ModelNotFoundError, ModelDisabledError, AmbiguousModelError) as exc:
            logger.error("Model registry resolution failed: %s", type(exc).__name__)
            raise LLMResponseException() from exc

        try:
            resolved = await self._resolution_coordinator.resolve(
                request_provider=catalog_model.provider.value,
                request_model=catalog_model.name,
            )
        except ProviderResolutionError as exc:
            logger.error("Provider resolution failed: %s", type(exc).__name__)
            raise LLMResponseException() from exc

        logger.info(
            "Selected provider=%s, model=%s (registry-validated)",
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
        """
        Instantiate the concrete provider adapter for a resolved context.

        Args:
            context: The resolved provider context, supplying the provider
                name to look up in the factory/registry and the kwargs to
                construct it with.

        Returns:
            A new :class:`~src.providers.base.BaseProvider` adapter instance.

        Raises:
            ProviderNotFoundError: If ``context.provider_name`` is not
                registered with the :class:`~src.providers.registry.ProviderRegistry`
                (propagated from :class:`~src.providers.factory.ProviderFactory`).
        """
        return self._factory.create(context.provider_name, **context.factory_kwargs)

    async def _execute_provider_call(
        self,
        context: ProviderContext,
        operation: Callable[[BaseProvider], Awaitable[ResultT]],
    ) -> ResultT:
        """
        Create a provider, run an operation against it, and normalize the outcome.

        Shared plumbing for :meth:`chat` and :meth:`check_connection`:
        creates the provider adapter, ensures it is properly closed (via
        :meth:`_run_with_provider`), measures and logs latency, and maps any
        :class:`~src.providers.exceptions.ProviderError` into the
        application's normalized exception types before it can reach the
        caller.

        Args:
            context: The resolved provider context for this call.
            operation: An async callable that receives the live provider
                adapter and returns the operation-specific result (e.g. the
                chat response dict, or the health-check dict). Parameterizing
                the actual provider call as a callback is what lets this
                method be shared between two otherwise-different operations.

        Returns:
            Whatever ``operation`` returns, unchanged.

        Raises:
            OllamaConnectionException: If the provider raises
                :class:`~src.providers.exceptions.ProviderUnavailableError`.
            LLMResponseException: If the provider raises any other
                :class:`~src.providers.exceptions.ProviderError` subclass.
        """
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
        """
        Run ``operation`` against ``provider``, closing the provider afterwards.

        Provider adapters may or may not implement the async context manager
        protocol (``__aenter__``/``__aexit__``) — some hold pooled
        connections that benefit from ``async with`` semantics, others expose
        only a plain ``close()`` method (or nothing to clean up at all). This
        method supports all three shapes uniformly so
        :meth:`_execute_provider_call` does not need to know which cleanup
        style a given adapter uses.

        The check specifically inspects
        ``provider_type.__dict__.get("__aenter__")`` rather than just
        ``hasattr(provider, "__aenter__")`` because ``ABC``/``object`` can
        cause ``hasattr`` to report false positives in some class hierarchies;
        checking the class's own ``__dict__`` confirms the method is actually
        defined on this concrete class rather than inherited as a stub.

        Args:
            provider: The live provider adapter to invoke.
            operation: The async callback to run with ``provider``.

        Returns:
            Whatever ``operation`` returns, unchanged.

        Raises:
            ProviderError: Propagated unmodified from ``operation`` or from
                provider cleanup; mapping to application exceptions happens
                one level up, in :meth:`_execute_provider_call`.
        """
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
            # Fall back to a plain close() for adapters that manage a
            # resource (e.g. an HTTP client) without implementing the full
            # async context manager protocol.
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    @staticmethod
    def _map_provider_error(exc: ProviderError) -> Exception:
        """
        Translate a provider-layer error into the application's normalized exception type.

        Only ``ProviderUnavailableError`` (backend unreachable) gets its own
        distinct exception (``OllamaConnectionException``, historically named
        for the original single-provider implementation but now used for any
        provider's connectivity failure) since routes/clients may want to
        react differently to "can't connect" versus other failures. Every
        other :class:`~src.providers.exceptions.ProviderError` subclass
        (authentication, rate limiting, invalid request, etc.) collapses to
        the generic ``LLMResponseException`` — deliberately coarse-grained
        for now, since the HTTP layer does not yet differentiate these cases
        (see ``08-security.mdc`` on returning generic error messages).

        Args:
            exc: The provider error raised by the adapter.

        Returns:
            An ``OllamaConnectionException`` for connectivity failures, or an
            ``LLMResponseException`` for any other provider error.

        Example:
            >>> AIService._map_provider_error(ProviderUnavailableError("down"))
            OllamaConnectionException(...)
            >>> AIService._map_provider_error(RateLimitError("too many requests"))
            LLMResponseException(...)
        """
        if isinstance(exc, ProviderUnavailableError):
            return OllamaConnectionException()
        return LLMResponseException()

    @staticmethod
    def _to_provider_type(provider_name: str) -> ProviderType:
        """
        Convert a resolved provider name string into the ``ProviderType`` enum.

        Args:
            provider_name: The resolved provider's stable machine key (e.g.
                ``"ollama"``), as produced by
                :class:`~src.services.provider_resolver.ProviderResolver`.

        Returns:
            The matching :class:`~src.core.enums.ProviderType` member.

        Raises:
            ValueError: If ``provider_name`` does not match any
                ``ProviderType`` member (e.g. a database provider name that
                predates the enum, or a typo).
        """
        return ProviderType(provider_name)


def create_ai_service(
    *,
    factory: ProviderFactory | None = None,
    registry: ProviderRegistry | None = None,
    settings: Settings | None = None,
    resolution_coordinator: ProviderResolutionCoordinator | None = None,
    model_registry_service: ModelRegistryService | None = None,
) -> AIService:
    """
    Build an :class:`AIService` with shared registry/factory defaults.

    This is the preferred way to construct an ``AIService`` in application
    code (e.g. FastAPI dependency providers): it wires up the process-wide
    provider registry and factory automatically, while still allowing every
    collaborator to be overridden — which is what makes this function useful
    both for production wiring and for unit tests that need to inject fakes.

    Args:
        factory: The provider factory to use. Defaults to a new
            :class:`~src.providers.factory.ProviderFactory` built from
            ``registry``.
        registry: The provider registry to use when building a default
            ``factory``. Defaults to the shared global registry via
            :func:`~src.providers.registry.get_registry`. Ignored if
            ``factory`` is explicitly supplied.
        settings: Application settings. Defaults to
            :func:`~src.core.config.get_settings`.
        resolution_coordinator: The resolution coordinator to use. Defaults
            to a new
            :class:`~src.services.provider_resolution_coordinator.ProviderResolutionCoordinator`
            backed by the application's shared ``AsyncSessionLocal`` session
            factory.
        model_registry_service: Catalog service for registry-only model
            selection. Defaults to an empty in-memory registry when omitted
            (primarily for legacy unit tests).

    Returns:
        A fully constructed :class:`AIService` ready to handle chat and
        health-check operations.

    Example:
        >>> ai_service = create_ai_service()
        >>> await ai_service.check_connection()
        {'status': <HealthStatus.HEALTHY: 'healthy'>, ...}
    """
    resolved_settings = settings or get_settings()
    resolved_registry = registry or get_registry()
    resolved_factory = factory or ProviderFactory(resolved_registry)
    coordinator = resolution_coordinator or ProviderResolutionCoordinator(
        session_factory=AsyncSessionLocal,
        settings=resolved_settings,
    )
    from src.registry.memory import MemoryModelRegistry

    registry_service = model_registry_service or ModelRegistryService(
        MemoryModelRegistry(),
        settings=resolved_settings,
    )
    return AIService(
        factory=resolved_factory,
        resolution_coordinator=coordinator,
        model_registry_service=registry_service,
        settings=resolved_settings,
    )
