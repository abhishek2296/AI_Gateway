"""
Unit tests for ``AIService`` using in-memory stub providers.

``AIService`` (see ``src/services/ai_service.py``) is the orchestration layer
that: resolves which provider/model to use (via a resolution coordinator),
constructs a provider client through ``ProviderFactory``, executes a chat
request, translates provider-level errors into the gateway's own exception
types, and ensures the provider client is always cleaned up afterwards.

These tests never talk to a real LLM backend — they register lightweight
``StubProvider``/``ClosableOnlyProvider`` test doubles (defined below) under
the ``"ollama"`` provider name and mock out the resolution coordinator, so
only ``AIService``'s own orchestration logic is under test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import Settings
from src.core.enums import ProviderType
from src.core.exceptions import LLMResponseException, OllamaConnectionException
from src.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthCheckResult,
    ModelInfo,
)
from src.providers.exceptions import ModelNotFoundError, ProviderUnavailableError
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry
from src.registry.memory import MemoryModelRegistry
from src.core.enums import ProviderType
from src.registry.models import ModelCapability, ModelInfo
from src.registry.read_only import ReadOnlyModelRegistryView
from src.services.ai_service import AIService
from src.services.model_registry_service import ModelRegistryService
from src.services.resolution_types import ResolutionSource, ResolvedConfiguration


class StubProvider(BaseProvider):
    """
    In-memory ``BaseProvider`` test double, registered under the "ollama"
    provider name so ``ProviderFactory``/``ProviderRegistry`` route to it.

    Records every chat request it receives (``chat_calls``) and every
    health check performed (``health_calls``) so tests can assert on
    ``AIService``'s call patterns, and supports the async context-manager
    protocol so cleanup-on-exit behavior can be verified.

    Attributes:
        chat_calls: All ``ChatRequest`` objects passed to ``chat()``, in
            call order — used to assert which model/provider ``AIService``
            actually routed a request to.
        health_calls: Number of times ``health_check()`` has been invoked.
        closed: Whether ``close()`` (directly or via ``__aexit__``) has run.

    Example:
        >>> provider = StubProvider()
        >>> provider.provider_name
        'ollama'
    """

    provider_name = "ollama"

    def __init__(self) -> None:
        self.chat_calls: list[ChatRequest] = []
        self.health_calls = 0
        self.closed = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Record the request and return a fixed, recognizable stub response."""
        self.chat_calls.append(request)
        return ChatResponse(content="stub-response", model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Yield a single stub chunk (streaming is not exercised by these tests)."""
        yield ChatStreamChunk(content="stub")

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Return a minimal, fixed embedding vector for any request."""
        return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])

    async def list_models(self) -> Sequence[ModelInfo]:
        """Return a single placeholder model entry."""
        return (ModelInfo(id="stub", name="stub"),)

    async def health_check(self) -> HealthCheckResult:
        """Record the call and report a fixed healthy status."""
        self.health_calls += 1
        return HealthCheckResult(healthy=True, latency_ms=12.5)

    async def close(self) -> None:
        """Mark this stub as closed so tests can assert cleanup happened."""
        self.closed = True

    async def __aenter__(self) -> StubProvider:
        """Support ``async with`` usage, mirroring real provider clients."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Ensure ``close()`` runs on context-manager exit, like a real client."""
        await self.close()


class ClosableOnlyProvider(BaseProvider):
    """
    Provider test double that exposes ``close()`` but *not* an async context
    manager (``__aenter__``/``__aexit__``).

    Exists specifically to verify ``AIService`` falls back to calling
    ``close()`` directly when a provider doesn't support ``async with``
    (see ``test_provider_cleanup_without_context_manager``), covering both
    supported cleanup styles a provider implementation might use.

    Attributes:
        closed: Whether ``close()`` has been called.
    """

    provider_name = "ollama"

    def __init__(self) -> None:
        self.closed = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Return a fixed, recognizable stub response (distinct from ``StubProvider``'s)."""
        return ChatResponse(content="closable-response", model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Yield a single stub chunk (streaming is not exercised by these tests)."""
        yield ChatStreamChunk(content="stub")

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Return a minimal, fixed embedding vector for any request."""
        return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])

    async def list_models(self) -> Sequence[ModelInfo]:
        """Return a single placeholder model entry."""
        return (ModelInfo(id="stub", name="stub"),)

    async def health_check(self) -> HealthCheckResult:
        """Report a fixed healthy status."""
        return HealthCheckResult(healthy=True, latency_ms=1.0)

    async def close(self) -> None:
        """Mark this stub as closed so tests can assert cleanup happened."""
        self.closed = True


def _settings(**overrides: object) -> Settings:
    """
    Build a ``Settings`` instance with sane test defaults for ``AIService`` tests.

    Args:
        **overrides: Field name/value pairs that override the defaults.

    Returns:
        A ``Settings`` object with ``DEFAULT_PROVIDER="ollama"`` and
        ``DEFAULT_MODEL="qwen3:8b"`` unless overridden.
    """
    defaults = {
        "DATABASE_URL": "postgresql+asyncpg://ai_app:ai_app_password@localhost:5432/ai_coding_assistant",
        "DEFAULT_PROVIDER": "ollama",
        "DEFAULT_MODEL": "qwen3:8b",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _resolved_config(
    *,
    provider_name: str = "ollama",
    model_name: str = "qwen3:8b",
    factory_kwargs: dict[str, object] | None = None,
) -> ResolvedConfiguration:
    """
    Build a ``ResolvedConfiguration`` as if produced by the resolution
    coordinator, to stub ``resolution_coordinator.resolve.return_value``.

    Args:
        provider_name: The provider ``AIService`` should route the request
            to, e.g. ``"ollama"``.
        model_name: The model name ``AIService`` should pass through to the
            provider's chat request.
        factory_kwargs: Keyword arguments ``AIService`` should forward to
            ``ProviderFactory.create``. Defaults to a representative
            Ollama base URL/timeout pair if not supplied.

    Returns:
        A ``ResolvedConfiguration`` with both ``provider_source`` and
        ``model_source`` set to ``ResolutionSource.SETTINGS`` (the specific
        source doesn't matter for ``AIService`` orchestration tests, only
        the resolved names/kwargs do).

    Example:
        >>> config = _resolved_config(model_name="custom-model")
        >>> config.model_name
        'custom-model'
    """
    return ResolvedConfiguration(
        provider_name=provider_name,
        model_name=model_name,
        factory_kwargs=factory_kwargs or {"base_url": "http://localhost:11434", "timeout": 60.0},
        provider_source=ResolutionSource.SETTINGS,
        model_source=ResolutionSource.SETTINGS,
    )


async def _seed_registry(
    registry: MemoryModelRegistry,
    *,
    name: str = "qwen3:8b",
    provider: ProviderType = ProviderType.OLLAMA,
    is_default: bool = True,
) -> None:
    await registry.register(
        ModelInfo(
            name=name,
            provider=provider,
            display_name=name,
            description="Test catalog entry.",
            capabilities=frozenset({ModelCapability.CHAT}),
            metadata={"is_default": True} if is_default else {},
        ),
    )


async def _make_model_registry_service_async(
    *,
    name: str = "qwen3:8b",
    provider: ProviderType = ProviderType.OLLAMA,
    is_default: bool = True,
) -> ModelRegistryService:
    registry = MemoryModelRegistry()
    await _seed_registry(registry, name=name, provider=provider, is_default=is_default)
    return ModelRegistryService(ReadOnlyModelRegistryView(registry), settings=_settings())


def _make_model_registry_service(
    *,
    name: str = "qwen3:8b",
    provider: ProviderType = ProviderType.OLLAMA,
    is_default: bool = True,
) -> ModelRegistryService:
    import asyncio

    registry = MemoryModelRegistry()

    async def seed() -> None:
        await _seed_registry(registry, name=name, provider=provider, is_default=is_default)

    asyncio.run(seed())
    return ModelRegistryService(ReadOnlyModelRegistryView(registry), settings=_settings())


@pytest.fixture
def stub_provider() -> StubProvider:
    """Fresh ``StubProvider`` per test, so call-recording state never leaks between tests."""
    return StubProvider()


@pytest.fixture
def resolution_coordinator() -> AsyncMock:
    """
    Mock resolution coordinator pre-configured to resolve to the default
    test provider/model.

    Returns:
        An ``AsyncMock`` whose ``resolve`` method returns
        ``_resolved_config()`` by default; individual tests may override
        ``resolution_coordinator.resolve.return_value`` to exercise a
        different provider/model/kwargs combination.
    """
    coordinator = AsyncMock()
    coordinator.resolve = AsyncMock(return_value=_resolved_config())
    return coordinator


@pytest.fixture
def ai_service(stub_provider: StubProvider, resolution_coordinator: AsyncMock) -> AIService:
    """
    Build an ``AIService`` wired to a ``StubProvider`` via a real
    ``ProviderRegistry``/``ProviderFactory``, with ``factory.create`` patched
    to always hand back the shared ``stub_provider`` instance.

    Patching ``factory.create`` (rather than only registering the class)
    lets tests assert against the *same* ``StubProvider`` object that
    ``AIService`` ends up using — e.g. inspecting ``stub_provider.chat_calls``
    after invoking ``ai_service.chat(...)``.

    Args:
        stub_provider: The ``StubProvider`` instance ``factory.create``
            will always return.
        resolution_coordinator: Mocked coordinator supplying the resolved
            provider/model/kwargs.

    Returns:
        An ``AIService`` ready to exercise in a test.
    """
    registry = ProviderRegistry()
    registry.register(StubProvider)
    factory = ProviderFactory(registry)
    factory.create = MagicMock(return_value=stub_provider)
    return AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        model_registry_service=_make_model_registry_service(),
        settings=_settings(),
    )


@pytest.mark.asyncio
async def test_chat_uses_default_provider_and_model(
    ai_service: AIService,
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    """
    Calling ``chat()`` with no explicit provider/model must ask the
    resolution coordinator to resolve using ``request_provider=None`` and
    ``request_model=None``, then use whatever it resolves to.

    Confirms both the coordinator is consulted with the right (empty)
    request-level overrides, and the resolved model/provider correctly flow
    through to the response and to the underlying provider call.
    """
    result = await ai_service.chat("hello")

    resolution_coordinator.resolve.assert_awaited_once_with(
        request_provider="ollama",
        request_model="qwen3:8b",
    )
    assert result["response"] == "stub-response"
    assert result["model"] == "qwen3:8b"
    assert result["provider"] == ProviderType.OLLAMA
    assert len(stub_provider.chat_calls) == 1
    assert stub_provider.chat_calls[0].model == "qwen3:8b"


@pytest.mark.asyncio
async def test_chat_honors_explicit_provider_and_model(
    ai_service: AIService,
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    """
    Explicit ``provider``/``model`` arguments to ``chat()`` must be forwarded
    unchanged to the resolution coordinator as ``request_provider``/
    ``request_model``.

    ``AIService`` itself doesn't decide precedence between request overrides
    and defaults — that's the resolver's job (see
    ``test_provider_resolution.py``) — but it must correctly pass through
    what the caller supplied and use whatever the coordinator resolves to.
    """
    resolution_coordinator.resolve.return_value = _resolved_config(model_name="custom-model")

    registry = MemoryModelRegistry()
    await _seed_registry(registry, name="qwen3:8b", is_default=True)
    await registry.register(
        ModelInfo(
            name="custom-model",
            provider=ProviderType.OLLAMA,
            display_name="custom-model",
            description="override",
            capabilities=frozenset({ModelCapability.CHAT}),
        ),
    )
    model_registry_service = ModelRegistryService(registry, settings=_settings())
    provider_registry = ProviderRegistry()
    provider_registry.register(StubProvider)
    factory = ProviderFactory(provider_registry)
    factory.create = MagicMock(return_value=stub_provider)
    ai_service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        model_registry_service=model_registry_service,
        settings=_settings(),
    )

    result = await ai_service.chat(
        "hello",
        provider="ollama",
        model="custom-model",
    )

    resolution_coordinator.resolve.assert_awaited_with(
        request_provider="ollama",
        request_model="custom-model",
    )
    assert result["model"] == "custom-model"
    assert stub_provider.chat_calls[0].model == "custom-model"


@pytest.mark.asyncio
async def test_chat_uses_resolved_model_from_coordinator(
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    """
    The final chat response's ``model`` field should reflect whatever model
    name the resolution coordinator resolved to, not a hardcoded default.

    Builds an ``AIService`` directly (rather than via the ``ai_service``
    fixture) to make the dependency wiring explicit and confirm nothing in
    ``AIService`` overrides or ignores the coordinator's resolved model
    name.
    """
    resolution_coordinator.resolve.return_value = _resolved_config(
        model_name="default-from-coordinator",
    )
    registry = ProviderRegistry()
    registry.register(StubProvider)
    factory = ProviderFactory(registry)
    factory.create = MagicMock(return_value=stub_provider)
    service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        model_registry_service=await _make_model_registry_service_async(
            name="default-from-coordinator",
        ),
        settings=_settings(),
    )

    result = await service.chat("hello")

    assert result["model"] == "default-from-coordinator"


@pytest.mark.asyncio
async def test_check_connection_returns_legacy_payload(ai_service: AIService) -> None:
    """
    ``check_connection()`` should translate the provider's
    ``HealthCheckResult`` into the legacy dict-shaped payload consumers
    (e.g. the ``/health`` route) expect: ``connected``, ``status``,
    ``provider``, ``model``, and ``latency_ms`` keys.

    This guards backward compatibility of the response shape while the
    underlying implementation was refactored to use the provider
    abstraction internally.
    """
    result = await ai_service.check_connection()

    assert result["connected"] is True
    assert result["status"].value == "healthy"
    assert result["provider"] == ProviderType.OLLAMA
    assert result["model"] == "qwen3:8b"
    assert result["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_provider_unavailable_maps_to_connection_exception(
    ai_service: AIService,
    stub_provider: StubProvider,
) -> None:
    """
    A ``ProviderUnavailableError`` raised by the provider client must be
    translated into the gateway's ``OllamaConnectionException``.

    Callers of ``AIService`` should only ever see the gateway's own
    exception hierarchy (``src/core/exceptions.py``), never a
    provider-specific exception type — this keeps error handling in routes
    provider-agnostic. ``stub_provider.chat`` is monkey-patched here to
    simulate the provider raising this specific error.
    """

    async def failing_chat(_request: ChatRequest) -> ChatResponse:
        raise ProviderUnavailableError("down", provider="ollama")

    stub_provider.chat = failing_chat  # type: ignore[method-assign]

    with pytest.raises(OllamaConnectionException):
        await ai_service.chat("hello")


@pytest.mark.asyncio
async def test_other_provider_errors_map_to_llm_response_exception(
    ai_service: AIService,
    stub_provider: StubProvider,
) -> None:
    """
    Provider errors other than "unavailable" (e.g. ``ModelNotFoundError``)
    must be translated into the generic ``LLMResponseException`` rather than
    the connection-specific exception.

    This confirms ``AIService``'s error-mapping distinguishes "the provider
    itself is down/unreachable" from "the provider responded but rejected
    this specific request" — the two cases warrant different exception
    types so callers/clients can react appropriately (e.g. retry-with-backoff
    vs. surface a client-facing 4xx).
    """

    async def failing_chat(_request: ChatRequest) -> ChatResponse:
        raise ModelNotFoundError("missing", provider="ollama")

    stub_provider.chat = failing_chat  # type: ignore[method-assign]

    with pytest.raises(LLMResponseException):
        await ai_service.chat("hello")


@pytest.mark.asyncio
async def test_provider_cleanup_on_context_manager(
    ai_service: AIService,
    stub_provider: StubProvider,
) -> None:
    """
    When the provider client supports the async context-manager protocol
    (``StubProvider`` defines ``__aenter__``/``__aexit__``), ``AIService``
    must use it, guaranteeing ``close()`` runs even without an explicit
    call — proven here by asserting ``stub_provider.closed`` after a normal,
    successful ``chat()`` call.
    """
    await ai_service.chat("hello")

    assert stub_provider.closed is True


@pytest.mark.asyncio
async def test_provider_cleanup_without_context_manager(
    resolution_coordinator: AsyncMock,
) -> None:
    """
    When the provider client does *not* implement the async context-manager
    protocol (``ClosableOnlyProvider`` only has ``close()``), ``AIService``
    must fall back to calling ``close()`` directly so resources are still
    released.

    This is the complementary case to
    ``test_provider_cleanup_on_context_manager``, confirming ``AIService``
    supports both provider cleanup styles.
    """
    provider = ClosableOnlyProvider()
    registry = ProviderRegistry()
    registry.register(ClosableOnlyProvider)
    factory = ProviderFactory(registry)
    factory.create = MagicMock(return_value=provider)
    service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        model_registry_service=await _make_model_registry_service_async(),
        settings=_settings(),
    )

    await service.chat("hello")

    assert provider.closed is True


@pytest.mark.asyncio
async def test_create_provider_uses_factory(
    stub_provider: StubProvider,
    resolution_coordinator: AsyncMock,
) -> None:
    """
    ``AIService`` must call ``ProviderFactory.create`` with the resolved
    provider name and exactly the ``factory_kwargs`` the resolution
    coordinator produced — no extra keys added, none dropped.

    This confirms the resolved configuration's ``factory_kwargs`` (e.g. a
    custom ``base_url``/``timeout`` for a non-default deployment) reach the
    provider factory unmodified, which is how per-request/per-provider
    configuration actually takes effect.
    """
    factory_kwargs = {"base_url": "http://custom:11434", "timeout": 30.0}
    resolution_coordinator.resolve.return_value = _resolved_config(factory_kwargs=factory_kwargs)
    registry = ProviderRegistry()
    registry.register(StubProvider)
    factory = ProviderFactory(registry)
    create_mock = MagicMock(return_value=stub_provider)
    factory.create = create_mock
    service = AIService(
        factory=factory,
        resolution_coordinator=resolution_coordinator,
        model_registry_service=await _make_model_registry_service_async(),
        settings=_settings(),
    )

    await service.chat("hello")

    create_mock.assert_called_once_with("ollama", **factory_kwargs)
