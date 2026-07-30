"""
Abstract LLM service contract for legacy, single-provider-style consumers.

``BaseLLMService`` predates the multi-provider gateway architecture described
in ``05-ai-gateway-architecture.mdc`` (compare with the richer
:class:`~src.providers.base.BaseProvider`, which is the current provider
abstraction used by :class:`~src.services.ai_service.AIService`). It is kept
around as a narrow, string-in/string-out interface for any code path that
still expects that simpler shape — currently
:class:`~src.services.ollama_service.OllamaService` (a direct, standalone
implementation) and :class:`~src.services.llm_adapter.LLMHealthAdapter` (which
adapts the modern ``AIService`` back down to this interface) both implement
it.

New provider integrations should implement
:class:`~src.providers.base.BaseProvider` instead; this contract exists only
for backward compatibility with call sites that have not yet migrated.
"""

from abc import ABC, abstractmethod


class BaseLLMService(ABC):
    """
    Minimal abstract contract for a chat-capable LLM backend.

    Any concrete subclass must implement both ``chat`` and
    ``check_connection``. This interface intentionally supports only a single
    plain-text prompt in and plain-text (or dict) response out — it has no
    concept of multi-turn history, streaming, tool calls, or provider/model
    overrides, unlike :class:`~src.providers.base.BaseProvider`.

    Example:
        >>> class EchoService(BaseLLMService):
        ...     async def chat(self, message: str) -> str:
        ...         return message
        ...     async def check_connection(self) -> dict:
        ...         return {"connected": True}
        >>> service = EchoService()
        >>> import asyncio
        >>> asyncio.run(service.chat("hi"))
        'hi'
    """

    @abstractmethod
    async def chat(self, message: str) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            message: The user's plain-text prompt.

        Returns:
            The LLM's plain-text reply.

        Raises:
            NotImplementedError: If called directly on ``BaseLLMService``
                rather than a concrete subclass (enforced by ``@abstractmethod``;
                Python raises ``TypeError`` at instantiation time instead if
                a subclass forgets to override this).
        """
        pass

    @abstractmethod
    async def check_connection(self) -> dict:
        """
        Probe backend connectivity/health.

        Returns:
            A ``dict`` describing connection health. Concrete subclasses
            define their own exact keys (e.g.
            :class:`~src.services.ollama_service.OllamaService` returns
            ``status``, ``provider``, ``model``, ``connected``,
            ``latency_ms``); this base contract only fixes the return type as
            ``dict``, not its schema.

        Raises:
            NotImplementedError: If called directly on ``BaseLLMService``
                rather than a concrete subclass.
        """
        pass