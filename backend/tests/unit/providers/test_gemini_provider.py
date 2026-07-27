"""Unit tests for GeminiProvider using respx mocks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ChatMessage, ChatRequest, EmbeddingsRequest, ResponseFormat
from src.providers.gemini import GeminiProvider

BASE_URL = "https://generativelanguage.googleapis.com"


@pytest.fixture
def gemini(respx_mock: respx.MockRouter) -> GeminiProvider:
    respx_mock.base_url = BASE_URL
    return GeminiProvider(api_key="test-key", base_url=BASE_URL)



@pytest.mark.asyncio
async def test_chat_success(gemini: GeminiProvider, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("/v1beta/models/gemini-pro:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hello"}]},
                        "finishReason": "STOP",
                    },
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            },
        ),
    )

    response = await gemini.chat(
        ChatRequest(
            model="gemini-pro",
            messages=[
                ChatMessage(role="system", content="Be helpful."),
                ChatMessage(role="user", content="Hi"),
            ],
            provider_options={
                "safetySettings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}],
            },
        ),
    )

    assert response.content == "Hello"
    assert response.finish_reason == "STOP"
    assert response.usage is not None
    assert response.usage.total_tokens == 7

    body = json.loads(route.calls.last.request.read())
    assert "systemInstruction" in body
    assert "safetySettings" in body
    assert route.calls.last.request.headers["x-goog-api-key"] == "test-key"



@pytest.mark.asyncio
async def test_stream_chat_sse(gemini: GeminiProvider, respx_mock: respx.MockRouter) -> None:
    stream_body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n'
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}]}\n\n'
    )
    respx_mock.post("/v1beta/models/gemini-pro:streamGenerateContent").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body.encode(),
        ),
    )

    chunks = []
    async for chunk in gemini.stream_chat(
        ChatRequest(model="gemini-pro", messages=[ChatMessage(role="user", content="Hi")]),
    ):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == ["Hel", "lo"]
    assert chunks[-1].finish_reason == "STOP"



@pytest.mark.asyncio
async def test_embeddings_success(gemini: GeminiProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1beta/models/text-embedding-004:embedContent").mock(
        return_value=httpx.Response(
            200,
            json={"embedding": {"values": [0.1, 0.2, 0.3]}},
        ),
    )

    response = await gemini.embeddings(
        EmbeddingsRequest(model="text-embedding-004", input="hello"),
    )

    assert response.embeddings == [[0.1, 0.2, 0.3]]



@pytest.mark.asyncio
async def test_list_models_and_validate(gemini: GeminiProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.get("/v1beta/models").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "models/gemini-pro"}, {"name": "models/text-embedding-004"}]},
        ),
    )

    models = await gemini.list_models()
    assert len(models) == 2
    assert await gemini.validate_model("gemini-pro") is True
    assert await gemini.validate_model("unknown") is False



@pytest.mark.asyncio
async def test_json_response_format_in_payload(
    gemini: GeminiProvider,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post("/v1beta/models/gemini-pro:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]},
        ),
    )

    await gemini.chat(
        ChatRequest(
            model="gemini-pro",
            messages=[ChatMessage(role="user", content="Return JSON")],
            response_format=ResponseFormat(type="json"),
        ),
    )

    body = json.loads(route.calls.last.request.read())
    assert body["generationConfig"]["responseMimeType"] == "application/json"



@pytest.mark.asyncio
async def test_health_check(gemini: GeminiProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.get("/v1beta/models").mock(return_value=httpx.Response(200, json={"models": []}))
    result = await gemini.health_check()
    assert result.healthy is True
