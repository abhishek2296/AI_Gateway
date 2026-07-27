from src.services.ai_service import AIService


class ChatService:
    """HTTP-facing chat orchestration backed by the provider-agnostic AIService."""

    def __init__(self, ai_service: AIService) -> None:
        self._ai_service = ai_service

    async def chat(self, message: str):
        return await self._ai_service.chat(message)
