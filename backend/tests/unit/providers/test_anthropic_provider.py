"""Unit tests for AnthropicProvider using respx mocks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ChatMessage, ChatRequest, EmbeddingsRequest, ToolDefinition
from src.providers.exceptions import AuthenticationError, UnsupportedCapabilityError
from src.providers.anthropic import AnthropicProvider

BASE_URL = "https://api.anthropic.test"


@pytest.fixture
def anthropic(respx_mock: respx.MockRouter) -> AnthropicProvider:
    respx_mock.base_url = BASE_URL
    return AnthropicProvider(api_key="test-key", base_url=BASE_URL)



@pytest.mark.asyncio
async def test_chat_success(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        ),
    )

    response = await anthropic.chat(
        ChatRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[
                ChatMessage(role="system", content="Be concise."),
                ChatMessage(role="user", content="Hi"),
            ],
        ),
    )

    assert response.content == "Hello"
    assert response.finish_reason == "end_turn"
    assert response.usage is not None
    assert response.usage.total_tokens == 16

    body = json.loads(route.calls.last.request.read())
    assert body["system"] == "Be concise."
    assert request_headers_have_anthropic_auth(respx_mock.calls.last.request)


def request_headers_have_anthropic_auth(request: httpx.Request) -> bool:
    return request.headers.get("x-api-key") == "test-key" and "anthropic-version" in request.headers



@pytest.mark.asyncio
async def test_chat_with_tool_use(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "claude-3-5-sonnet-20241022",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"query": "weather"},
                    },
                ],
                "stop_reason": "tool_use",
            },
        ),
    )

    response = await anthropic.chat(
        ChatRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[ChatMessage(role="user", content="lookup weather")],
            tools=[ToolDefinition(name="lookup", parameters={"type": "object"})],
        ),
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "lookup"



@pytest.mark.asyncio
async def test_stream_chat_yields_text_deltas(
    anthropic: AnthropicProvider,
    respx_mock: respx.MockRouter,
) -> None:
    stream_body = (
        "event: content_block_delta\n"
        f"data: {json.dumps({'delta': {'type': 'text_delta', 'text': 'Hel'}})}\n\n"
        "event: content_block_delta\n"
        f"data: {json.dumps({'delta': {'type': 'text_delta', 'text': 'lo'}})}\n\n"
        "event: message_delta\n"
        f"data: {json.dumps({'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}})}\n\n"
    )
    respx_mock.post("/v1/messages").mock(
        return_value=httpx.Response(200, content=stream_body.encode()),
    )

    chunks = []
    async for chunk in anthropic.stream_chat(
        ChatRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[ChatMessage(role="user", content="Hi")],
        ),
    ):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks[:2]] == ["Hel", "lo"]
    assert chunks[-1].finish_reason == "end_turn"



@pytest.mark.asyncio
async def test_embeddings_raises_unsupported(anthropic: AnthropicProvider) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="embeddings"):
        await anthropic.embeddings(EmbeddingsRequest(model="claude", input="hello"))



@pytest.mark.asyncio
async def test_list_models(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.get("/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"}]},
        ),
    )

    models = await anthropic.list_models()
    assert models[0].supports_tools is True
    assert await anthropic.validate_model("claude-3-5-sonnet-20241022") is True



@pytest.mark.asyncio
async def test_health_check_and_auth_error(
    anthropic: AnthropicProvider,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("/v1/models").mock(return_value=httpx.Response(401))
    unhealthy = await anthropic.health_check()
    assert unhealthy.healthy is False

    respx_mock.get("/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    healthy = await anthropic.health_check()
    assert healthy.healthy is True



@pytest.mark.asyncio
async def test_chat_auth_error_maps(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/messages").mock(return_value=httpx.Response(401))

    with pytest.raises(AuthenticationError):
        await anthropic.chat(
            ChatRequest(
                model="claude-3-5-sonnet-20241022",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )
