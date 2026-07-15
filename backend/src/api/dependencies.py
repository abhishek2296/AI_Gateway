from fastapi import Depends
from src.services.base_llm import BaseLLMService
from src.services.chat_service import ChatService
from src.services.ollama_service import OllamaService


def get_llm_service() -> BaseLLMService:
    return OllamaService()

def get_chat_service(
    llm: BaseLLMService = Depends(get_llm_service),
) -> ChatService:

    return ChatService(llm)

def get_llm_service() -> BaseLLMService:
    return OllamaService()