"""Unit tests for OpenAIProvider using respx mocks."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.providers.base import (
    ChatMessage,
    ChatRequest,
    EmbeddingsRequest,
    ImagePart,
    ResponseFormat,
    ToolDefinition,
)
from src.providers.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from src.providers.openai import OpenAIProvider

BASE_URL = "https://api.openai.test"


@pytest.fixture
def openai(respx_mock: respx.MockRouter) -> OpenAIProvider:
    respx_mock.base_url = BASE_URL
    return OpenAIProvider(api_key="test-key", base_url=f"{BASE_URL}/v1")



@pytest.mark.asyncio
async def test_chat_success(openai: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ),
    )

    response = await openai.chat(
        ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.2,
            max_tokens=50,
        ),
    )

    assert response.content == "Hello"
    assert response.model == "gpt-4o"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert response.provider_response_id == "chatcmpl-1"

    request = respx_mock.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthenticationError),
        (404, ModelNotFoundError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
    ],
)
async def test_chat_http_errors(
    openai: OpenAIProvider,
    respx_mock: respx.MockRouter,
    status: int,
    expected: type[Exception],
) -> None:
    respx_mock.post("/v1/chat/completions").mock(return_value=httpx.Response(status))

    with pytest.raises(expected):
        await openai.chat(
            ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="Hi")]),
        )



@pytest.mark.asyncio
async def test_stream_chat_yields_chunks(
    openai: OpenAIProvider,
    respx_mock: respx.MockRouter,
) -> None:
    stream_body = (
        'data: {"choices":[{"delta":{"content":"Hel"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
        "data: [DONE]\n\n"
    )
    respx_mock.post("/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=stream_body.encode()),
    )

    chunks = []
    async for chunk in openai.stream_chat(
        ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="Hi")]),
    ):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == ["Hel", "lo", ""]
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None



@pytest.mark.asyncio
async def test_chat_with_tools_and_vision(openai: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
            },
        ),
    )

    response = await openai.chat(
        ChatRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(
                    role="user",
                    content="What is this?",
                    parts=[ImagePart(base64_data="abc123", media_type="image/png")],
                ),
            ],
            tools=[ToolDefinition(name="get_weather", parameters={"type": "object"})],
            response_format=ResponseFormat(type="json"),
        ),
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "get_weather"
    body = route.calls.last.request.read()
    assert b"image_url" in body
    assert b"tools" in body



@pytest.mark.asyncio
async def test_embeddings_success(openai: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [{"embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        ),
    )

    response = await openai.embeddings(
        EmbeddingsRequest(model="text-embedding-3-small", input="hello"),
    )

    assert response.embeddings == [[0.1, 0.2]]
    assert response.model == "text-embedding-3-small"



@pytest.mark.asyncio
async def test_list_models_and_validate_model(
    openai: OpenAIProvider,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3-small"}]},
        ),
    )

    models = await openai.list_models()
    assert len(models) == 2
    assert await openai.validate_model("gpt-4o") is True
    assert await openai.validate_model("missing-model") is False



@pytest.mark.asyncio
async def test_health_check_healthy(openai: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.get("/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o"}]}),
    )

    result = await openai.health_check()

    assert result.healthy is True
    assert result.models_available == 1



@pytest.mark.asyncio
async def test_health_check_unhealthy_on_auth_failure(
    openai: OpenAIProvider,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("/v1/models").mock(return_value=httpx.Response(401))

    result = await openai.health_check()

    assert result.healthy is False
