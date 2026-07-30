"""
Unit tests for `GeminiProvider` (`src/providers/gemini.py`) using respx mocks.

Covers Gemini's `generateContent`/`streamGenerateContent` request/response
mapping (including system instructions, provider-specific `safetySettings`
passthrough, and JSON response format), embeddings, model listing, and
health checks. All HTTP calls are intercepted with `respx` -- no real
Gemini API is contacted.
"""

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
    """Return a `GeminiProvider` with a fake API key, pointed at a base URL `respx_mock` intercepts."""
    respx_mock.base_url = BASE_URL
    return GeminiProvider(api_key="test-key", base_url=BASE_URL)



@pytest.mark.asyncio
async def test_chat_success(gemini: GeminiProvider, respx_mock: respx.MockRouter) -> None:
    """
    A successful chat call maps Gemini's `candidates`/`usageMetadata` response into `ChatResponse`.

    Also confirms three Gemini-specific request-shaping behaviors: a
    `system`-role message is hoisted into a top-level `systemInstruction`
    field (Gemini has no system role in its `contents` array), arbitrary
    `provider_options` (here `safetySettings`) are passed through verbatim
    into the request body, and Gemini's `x-goog-api-key` auth header is used
    instead of a bearer token.
    """
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
    """
    Streaming chat yields text from each SSE `data:` payload's `candidates` field, ending with `finishReason`.

    Gemini's streaming endpoint (`streamGenerateContent`) returns
    `content-type: text/event-stream`, so the mocked response explicitly
    sets that header to confirm the provider correctly recognizes and parses
    it as SSE rather than a plain JSON body.
    """
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
    """
    A successful embeddings call maps Gemini's single `embedding.values` response into a normalized `EmbeddingsResponse`.

    Unlike OpenAI/Ollama (which return a list of embedding vectors), Gemini's
    `embedContent` endpoint returns exactly one embedding under
    `embedding.values` -- the provider must wrap it in a single-element list
    to satisfy the shared `EmbeddingsResponse.embeddings: list[list[float]]` shape.
    """
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
    """
    `list_models()` strips Gemini's `"models/"` name prefix, and `validate_model()` checks against the stripped ids.

    Gemini's API returns model names as `"models/gemini-pro"` rather than
    the bare `"gemini-pro"` used everywhere else (request payloads, other
    providers' ids) -- the provider must normalize this so
    `validate_model("gemini-pro")` (the bare id callers actually use) works.
    """
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
    """
    `ResponseFormat(type="json")` is translated into Gemini's `generationConfig.responseMimeType`.

    Gemini has no generic `response_format` field like OpenAI -- structured
    output is requested via a MIME type nested under `generationConfig`, so
    this confirms the adapter performs that vendor-specific translation
    rather than sending an OpenAI-shaped field Gemini would ignore.
    """
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
    """A successful `/v1beta/models` call reports the provider healthy, even with an empty model list."""
    respx_mock.get("/v1beta/models").mock(return_value=httpx.Response(200, json={"models": []}))
    result = await gemini.health_check()
    assert result.healthy is True
