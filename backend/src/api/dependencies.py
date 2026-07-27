from functools import lru_cache

from fastapi import Depends

from src.providers.factory import ProviderFactory
from src.providers.registry import get_registry
from src.services.ai_service import AIService, create_ai_service
from src.services.base_llm import BaseLLMService
from src.services.chat_service import ChatService
from src.services.llm_adapter import LLMHealthAdapter


@lru_cache
def get_provider_factory() -> ProviderFactory:
    return ProviderFactory(get_registry())


@lru_cache
def get_ai_service() -> AIService:
    return create_ai_service(factory=get_provider_factory())


def get_chat_service(
    ai_service: AIService = Depends(get_ai_service),
) -> ChatService:
    return ChatService(ai_service)


def get_llm_service(
    ai_service: AIService = Depends(get_ai_service),
) -> BaseLLMService:
    return LLMHealthAdapter(ai_service)
