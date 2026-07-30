"""
Backward-compatible adapter for legacy :class:`~src.services.base_llm.BaseLLMService` consumers.

This module implements the Adapter pattern to bridge two generations of the
gateway's service layer: it lets code that was written against the narrow,
pre-multi-provider :class:`~src.services.base_llm.BaseLLMService` interface
keep working unchanged while the actual work is performed by the current
provider-agnostic :class:`~src.services.ai_service.AIService`. This avoids
forcing a simultaneous rewrite of every ``BaseLLMService`` call site when the
gateway's provider architecture evolved.
"""

from __future__ import annotations

from src.services.ai_service import AIService
from src.services.base_llm import BaseLLMService


class LLMHealthAdapter(BaseLLMService):
    """
    Delegates health and chat probes to :class:`AIService` without route changes.

    Wraps an :class:`~src.services.ai_service.AIService` instance and exposes
    it through the legacy :class:`~src.services.base_llm.BaseLLMService`
    interface, translating between the two contracts' slightly different
    return shapes (e.g. ``chat`` here returns a plain ``str``, while
    ``AIService.chat`` returns a structured ``dict``).

    Example:
        >>> adapter = LLMHealthAdapter(ai_service)
        >>> await adapter.chat("Hello!")
        'Hi there!'
        >>> await adapter.check_connection()
        {'status': <HealthStatus.HEALTHY: 'healthy'>, ...}
    """

    def __init__(self, ai_service: AIService) -> None:
        """
        Store the modern application service this adapter delegates to.

        Args:
            ai_service: The :class:`~src.services.ai_service.AIService`
                instance that actually performs provider resolution and
                invocation.
        """
        self._ai_service = ai_service

    async def chat(self, message: str) -> str:
        """
        Send a chat message and return only the reply text.

        Unwraps :meth:`AIService.chat`'s structured ``dict`` result down to
        just the ``response`` string, matching the narrower return type
        required by :meth:`~src.services.base_llm.BaseLLMService.chat`.

        Args:
            message: The user's plain-text prompt.

        Returns:
            The provider's reply text (model and provider metadata from the
            underlying call are discarded).

        Raises:
            OllamaConnectionException: When the resolved provider backend is
                unreachable.
            LLMResponseException: For other provider or resolution failures.

        Example:
            >>> await adapter.chat("What's 2+2?")
            '4'
        """
        result = await self._ai_service.chat(message)
        return str(result["response"])

    async def check_connection(self) -> dict:
        """
        Probe provider health via the underlying :class:`AIService`.

        Unlike :meth:`chat`, no translation is needed here since
        :meth:`AIService.check_connection` already returns a ``dict`` shape
        compatible with :meth:`~src.services.base_llm.BaseLLMService.check_connection`.

        Returns:
            The health-check ``dict`` produced by
            :meth:`AIService.check_connection` (``status``, ``provider``,
            ``model``, ``connected``, ``latency_ms``).

        Raises:
            LLMResponseException: If provider/model resolution fails before
                the health probe can be attempted.
        """
        return await self._ai_service.check_connection()
