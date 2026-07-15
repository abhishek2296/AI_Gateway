import asyncio

from src.services.ollama_service import OllamaService


async def main():
    service = OllamaService()

    response = await service.chat("Hello")

    print(response)


if __name__ == "__main__":
    asyncio.run(main())