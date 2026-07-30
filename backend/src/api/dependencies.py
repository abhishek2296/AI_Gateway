"""
FastAPI dependency-injection wiring for the API layer.

Centralizes how route handlers obtain their service-layer collaborators
(``ChatService``, ``BaseLLMService``) so routes never construct these objects
themselves — per the Layer Boundaries rule, ``api/routes/`` must not reach
into ``services/`` construction details directly. Each provider function
below is designed to be passed to FastAPI's ``Depends(...)``.

Two different caching strategies are used deliberately:

- ``get_provider_factory`` / ``get_ai_service`` are wrapped in
  ``functools.lru_cache`` because they are expensive to build (they wire up
  the provider registry) and hold no per-request state, so one shared
  instance can safely serve every request in the process.
- ``get_chat_service`` / ``get_llm_service`` are *not* cached — they are
  cheap, stateless wrapper objects, and FastAPI will happily construct a new
  one per request while still reusing the cached ``AIService`` underneath
  via nested ``Depends``.
"""

from functools import lru_cache

from fastapi import Depends, Request

from src.providers.factory import ProviderFactory
from src.providers.registry import get_registry
from src.registry.base import BaseModelRegistry
from src.registry.memory import MemoryModelRegistry
from src.services.ai_service import AIService, create_ai_service
from src.services.base_llm import BaseLLMService
from src.services.chat_service import ChatService
from src.services.llm_adapter import LLMHealthAdapter
from src.services.model_registry_service import ModelRegistryService

_ai_service_instance: AIService | None = None


def set_ai_service_instance(service: AIService) -> None:
    """Store the process-wide AIService built during application startup."""
    global _ai_service_instance
    _ai_service_instance = service


def get_model_registry(request: Request) -> BaseModelRegistry:
    """Return the catalog registry stored on ``app.state`` during startup."""
    return request.app.state.model_registry


def get_model_registry_service(request: Request) -> ModelRegistryService:
    """Return the registry service stored on ``app.state`` during startup."""
    return request.app.state.model_registry_service


@lru_cache
def get_provider_factory() -> ProviderFactory:
    """
    Build (once) the process-wide factory used to instantiate LLM providers.

    Cached with ``lru_cache`` (no arguments, so effectively a singleton)
    because ``ProviderFactory`` only needs the shared provider registry and
    has no per-request state — rebuilding it on every request would be
    wasted work with no benefit.

    Returns:
        A ``ProviderFactory`` bound to the global provider registry
        returned by ``providers.registry.get_registry``.

    Example:
        >>> factory_a = get_provider_factory()
        >>> factory_b = get_provider_factory()
        >>> factory_a is factory_b
        True
    """
    return ProviderFactory(get_registry())


@lru_cache
def get_ai_service() -> AIService:
    """
    Return the process-wide :class:`AIService` wired during application startup.

    Falls back to a fresh in-memory registry service when startup wiring has
    not run (e.g. some unit tests import dependencies without lifespan).
    """
    if _ai_service_instance is not None:
        return _ai_service_instance
    fallback_registry = MemoryModelRegistry()
    fallback_service = ModelRegistryService(fallback_registry)
    return create_ai_service(
        factory=get_provider_factory(),
        model_registry_service=fallback_service,
    )


def get_chat_service(
    ai_service: AIService = Depends(get_ai_service),
) -> ChatService:
    """
    Provide a :class:`ChatService` for the ``POST /chat`` route.

    Not cached: unlike ``AIService``, ``ChatService`` is a thin, stateless
    wrapper (see ``services/chat_service.py``) so there is no cost saved by
    reusing an instance across requests, and creating a fresh one keeps this
    dependency simple and free of shared mutable state.

    Args:
        ai_service: Injected by FastAPI via nested ``Depends(get_ai_service)``
            — resolves to the single cached ``AIService`` instance.

    Returns:
        A new ``ChatService`` wrapping the shared ``ai_service``.

    Example:
        >>> chat_service = get_chat_service(get_ai_service())
        >>> isinstance(chat_service, ChatService)
        True
    """
    return ChatService(ai_service)


def get_llm_service(
    ai_service: AIService = Depends(get_ai_service),
) -> BaseLLMService:
    """
    Provide a :class:`BaseLLMService` for the ``GET /health`` route.

    Wraps the shared ``AIService`` in ``LLMHealthAdapter`` so the health
    route can depend on the stable, pre-existing ``BaseLLMService`` contract
    (``chat`` / ``check_connection``) without the route needing to change
    when the underlying service implementation evolves — this is the
    adapter pattern referenced in ``services/llm_adapter.py``.

    Args:
        ai_service: Injected by FastAPI via nested ``Depends(get_ai_service)``
            — resolves to the single cached ``AIService`` instance.

    Returns:
        A new ``LLMHealthAdapter`` (typed as ``BaseLLMService``) delegating
        to the shared ``ai_service``.

    Example:
        >>> llm_service = get_llm_service(get_ai_service())
        >>> isinstance(llm_service, BaseLLMService)
        True
    """
    return LLMHealthAdapter(ai_service)
