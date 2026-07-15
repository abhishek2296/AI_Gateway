from abc import ABC, abstractmethod


class BaseLLMService(ABC):

    @abstractmethod
    async def chat(self, message: str) -> str:
        """
        Send a prompt to the LLM and return the response.
        """
        pass

    @abstractmethod
    async def check_connection(self) -> dict:
        pass