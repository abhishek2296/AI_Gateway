"""Backward-compatible adapter for legacy BaseLLMService consumers."""

from __future__ import annotations

from src.services.ai_service import AIService
from src.services.base_llm import BaseLLMService


class LLMHealthAdapter(BaseLLMService):
    """Delegates health and chat probes to :class:`AIService` without route changes."""

    def __init__(self, ai_service: AIService) -> None:
        self._ai_service = ai_service

    async def chat(self, message: str) -> str:
        result = await self._ai_service.chat(message)
        return str(result["response"])

    async def check_connection(self) -> dict:
        return await self._ai_service.check_connection()
