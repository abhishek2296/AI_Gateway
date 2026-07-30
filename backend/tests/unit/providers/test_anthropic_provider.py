"""
Unit tests for `AnthropicProvider` (`src/providers/anthropic.py`) using respx mocks.

Covers Anthropic Messages API request/response mapping (including system
prompt extraction, tool use, and SSE streaming with paired `event:`/`data:`
lines), model listing, health checks, and HTTP error mapping. Also confirms
that `embeddings()` is deliberately unsupported, since Anthropic's API has
no embeddings endpoint. All HTTP calls are intercepted with `respx` -- no
real Anthropic API is contacted.
"""

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
    """Return an `AnthropicProvider` with a fake API key, pointed at a base URL `respx_mock` intercepts."""
    respx_mock.base_url = BASE_URL
    return AnthropicProvider(api_key="test-key", base_url=BASE_URL)



@pytest.mark.asyncio
async def test_chat_success(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    """
    A successful chat call maps Anthropic's response into `ChatResponse`, and a `system` message is hoisted into the top-level `system` field.

    Anthropic's Messages API (unlike OpenAI/Ollama) does not accept a
    `system`-role entry inside the `messages` array -- it must be lifted out
    into a separate top-level `system` request field. This test confirms
    that translation happens, plus that vendor-specific auth headers
    (`x-api-key`, `anthropic-version`) are sent instead of a bearer token.
    """
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
    """
    Return whether `request` carries Anthropic's required auth headers.

    Anthropic authenticates via `x-api-key` (not a bearer token) and also
    requires an `anthropic-version` header pinning the API version being
    targeted; this helper centralizes that check for reuse across tests.

    Args:
        request: The outgoing `httpx.Request` captured from a `respx` mock call.

    Returns:
        `True` if both the API key and version headers are present/correct.

    Example:
        >>> req = httpx.Request(
        ...     "POST", "https://api.anthropic.test/v1/messages",
        ...     headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        ... )
        >>> request_headers_have_anthropic_auth(req)
        True
    """
    return request.headers.get("x-api-key") == "test-key" and "anthropic-version" in request.headers



@pytest.mark.asyncio
async def test_chat_with_tool_use(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    """
    A response with a `tool_use` content block is parsed into `response.tool_calls`.

    Anthropic represents tool invocations as a distinct content-block type
    (`"type": "tool_use"`) rather than a separate `tool_calls` field like
    OpenAI does, so this exercises Anthropic's specific parsing path.
    """
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
    """
    Streaming chat yields text from `content_block_delta` events and the finish reason from `message_delta`.

    Anthropic's stream is SSE with paired `event:`/`data:` lines (unlike
    OpenAI's plain `data:`-only SSE), so the two event types must be
    distinguished: `content_block_delta` carries incremental text, while
    `message_delta` carries the final `stop_reason` and usage totals.
    """
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
    """
    Calling `embeddings()` on `AnthropicProvider` raises `UnsupportedCapabilityError`, not a generic failure.

    Anthropic has no embeddings API. Raising a specific, well-named
    exception (rather than e.g. `NotImplementedError` or letting an HTTP
    call 404) lets callers detect "this provider doesn't support this
    capability" and route the request elsewhere instead of treating it as an
    unexpected error.
    """
    with pytest.raises(UnsupportedCapabilityError, match="embeddings"):
        await anthropic.embeddings(EmbeddingsRequest(model="claude", input="hello"))



@pytest.mark.asyncio
async def test_list_models(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    """
    `list_models()` maps Anthropic's model list into `ModelInfo`, flagging every Claude model as tool-capable.

    All current Claude models support tool use, so `supports_tools` should
    always be `True` here (unlike e.g. OpenAI, where capability can vary by
    model).
    """
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
    """
    `health_check()` toggles between unhealthy and healthy as the mocked response changes.

    Exercised as a single before/after sequence (rather than two separate
    tests) to confirm the provider doesn't cache a stale health result --
    each call must re-probe the endpoint and reflect its current response.
    """
    respx_mock.get("/v1/models").mock(return_value=httpx.Response(401))
    unhealthy = await anthropic.health_check()
    assert unhealthy.healthy is False

    respx_mock.get("/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    healthy = await anthropic.health_check()
    assert healthy.healthy is True



@pytest.mark.asyncio
async def test_chat_auth_error_maps(anthropic: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
    """A 401 from the Anthropic Messages API is mapped to `AuthenticationError`, not a raw HTTP error."""
    respx_mock.post("/v1/messages").mock(return_value=httpx.Response(401))

    with pytest.raises(AuthenticationError):
        await anthropic.chat(
            ChatRequest(
                model="claude-3-5-sonnet-20241022",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )
