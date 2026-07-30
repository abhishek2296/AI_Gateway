"""
Route-facing façade for the chat use case.

``ChatService`` sits directly behind the HTTP ``/chat`` route in the gateway's
layer flow (Route -> Service -> Provider). It exists so route handlers depend
on a small, chat-specific interface rather than reaching into the full
:class:`~src.services.ai_service.AIService` surface (which also handles
health checks and provider/model overrides) — keeping the route's dependency
narrow and making it easy to mock in route-level tests.
"""

from src.services.ai_service import AIService


class ChatService:
    """
    HTTP-facing chat orchestration backed by the provider-agnostic AIService.

    This is a thin façade: it holds no resolution or provider logic itself
    and simply forwards to :class:`~src.services.ai_service.AIService`, which
    performs provider/model resolution (via
    :class:`~src.services.provider_resolution_coordinator.ProviderResolutionCoordinator`)
    and the actual provider call.

    Attributes:
        _ai_service: The underlying application service that performs
            resolution and provider invocation.

    Example:
        >>> chat_service = ChatService(ai_service)
        >>> result = await chat_service.chat("Hello!")
        >>> result["response"]
        'Hi there! How can I help you today?'
    """

    def __init__(self, ai_service: AIService) -> None:
        """
        Store the application service this façade delegates to.

        Args:
            ai_service: The :class:`~src.services.ai_service.AIService`
                instance that performs provider resolution and the actual
                chat call.
        """
        self._ai_service = ai_service

    async def chat(
        self,
        message: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ):
        """
        Send a chat message using the model registry for provider/model selection.

        Args:
            message: The user's chat message text.
            model: Optional model name override (must exist in the registry).
            provider: Optional provider override (required when the model name
                is ambiguous across providers).

        Returns:
            Normalized chat result dict with ``response``, ``model``, ``provider``.
        """
        provider_value = provider.value if hasattr(provider, "value") else provider
        return await self._ai_service.chat(
            message,
            model=model,
            provider=provider_value,
        )
