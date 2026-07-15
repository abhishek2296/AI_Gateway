from src.services.base_llm import BaseLLMService


class ChatService:

    def __init__(self, llm: BaseLLMService):
        self.llm = llm

    async def chat(self, message: str):

        return await self.llm.chat(message)