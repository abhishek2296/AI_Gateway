"""
Unit tests for `OllamaProvider` (`src/providers/ollama.py`) using respx mocks.

Covers request/response mapping for chat (including streaming), embeddings,
health checks, and model listing against Ollama's native JSON API, plus how
HTTP failures (404/500/timeout/connection error) are translated into the
gateway's provider exception hierarchy so callers never see raw `httpx`
exceptions. All HTTP calls are intercepted with `respx` -- no real Ollama
server is contacted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from src.providers.base import (
    ChatMessage,
    ChatRequest,
    EmbeddingsRequest,
)
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
)
from src.providers.ollama import OllamaProvider

BASE_URL = "http://ollama.test"


@pytest.fixture
def ollama(respx_mock: respx.MockRouter) -> OllamaProvider:
    """Return an `OllamaProvider` pointed at a fake base URL intercepted by `respx_mock`."""
    respx_mock.base_url = BASE_URL
    return OllamaProvider(base_url=BASE_URL)


@respx.mock
@pytest.mark.asyncio
async def test_chat_success(ollama: OllamaProvider, respx_mock: respx.MockRouter) -> None:
    """
    A successful non-streaming chat call maps Ollama's response fields into `ChatResponse`.

    Also verifies the *outgoing* request body: Ollama's non-streaming API
    requires an explicit `"stream": false` flag, and `temperature`/`max_tokens`
    must be translated into Ollama's own `temperature`/`num_predict` option
    names -- getting either wrong would silently produce wrong provider
    behavior (e.g. an accidental streaming response) rather than an error.
    """
    respx_mock.post("/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "Hello there"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        ),
    )

    response = await ollama.chat(
        ChatRequest(
            model="qwen3:8b",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.7,
            max_tokens=100,
        ),
    )

    assert response.content == "Hello there"
    assert response.model == "qwen3:8b"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 16

    request = respx_mock.calls.last.request
    body = request.read()
    # Accept both spaced and compact JSON serialization: the assertion cares
    # about which fields/values were sent, not the exact whitespace style of
    # whichever JSON encoder produced the request body.
    assert b'"stream": false' in body or b'"stream":false' in body
    assert b'"temperature": 0.7' in body or b'"temperature":0.7' in body
    assert b'"num_predict": 100' in body or b'"num_predict":100' in body


@respx.mock
@pytest.mark.asyncio
async def test_stream_chat_yields_incremental_chunks(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """
    Streaming chat yields one chunk per NDJSON line, and the final chunk carries the finish reason.

    Ollama streams newline-delimited JSON objects rather than SSE; the last
    line (`"done":true`) carries `done_reason`/token-count fields that must
    surface on the terminal `ChatStreamChunk.finish_reason`, so callers can
    tell when the stream has ended and why.
    """
    stream_body = (
        '{"model":"qwen3:8b","message":{"role":"assistant","content":"Hel"},"done":false}\n'
        '{"model":"qwen3:8b","message":{"role":"assistant","content":"lo"},"done":false}\n'
        '{"model":"qwen3:8b","message":{"role":"assistant","content":""},'
        '"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}\n'
    )

    respx_mock.post("/api/chat").mock(
        return_value=httpx.Response(200, content=stream_body.encode()),
    )

    request = ChatRequest(
        model="qwen3:8b",
        messages=[ChatMessage(role="user", content="Hi")],
    )
    chunks: list[str] = []
    finish_reason: str | None = None
    async for chunk in ollama.stream_chat(request):
        chunks.append(chunk.content)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    assert chunks == ["Hel", "lo", ""]
    assert finish_reason == "stop"


@respx.mock
@pytest.mark.asyncio
async def test_embeddings_success(ollama: OllamaProvider, respx_mock: respx.MockRouter) -> None:
    """A successful embeddings call maps Ollama's response into a normalized `EmbeddingsResponse`."""
    respx_mock.post("/api/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3]],
                "prompt_eval_count": 8,
            },
        ),
    )

    response = await ollama.embeddings(
        EmbeddingsRequest(model="nomic-embed-text", input="hello"),
    )

    assert response.model == "nomic-embed-text"
    assert response.embeddings == [[0.1, 0.2, 0.3]]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 8


@respx.mock
@pytest.mark.asyncio
async def test_health_check_healthy(ollama: OllamaProvider, respx_mock: respx.MockRouter) -> None:
    """A reachable Ollama server reports healthy, with model count and latency populated."""
    respx_mock.get("/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]},
        ),
    )

    result = await ollama.health_check()

    assert result.healthy is True
    assert result.models_available == 1
    assert result.latency_ms is not None
    assert result.message == "Ollama is reachable."


@respx.mock
@pytest.mark.asyncio
async def test_health_check_unhealthy_on_connection_failure(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """
    `health_check()` reports unhealthy (not an exception) when Ollama is unreachable.

    Health checks must degrade gracefully rather than raising, since callers
    (e.g. a `/health` endpoint or startup probe) need a boolean result they
    can report even when the underlying service is completely down.
    """
    respx_mock.get("/api/tags").mock(side_effect=httpx.ConnectError("connection refused"))

    result = await ollama.health_check()

    assert result.healthy is False
    assert result.message == "Unable to reach Ollama."


@respx.mock
@pytest.mark.asyncio
async def test_list_models(ollama: OllamaProvider, respx_mock: respx.MockRouter) -> None:
    """
    `list_models()` maps Ollama's `/api/tags` entries into normalized `ModelInfo` objects.

    Also confirms `supports_streaming` defaults to `True` for Ollama models,
    since every Ollama chat model supports streaming responses natively.
    """
    respx_mock.get("/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:8b",
                        "model": "qwen3:8b",
                        "modified_at": "2026-01-01T00:00:00Z",
                        "size": 123,
                    },
                ],
            },
        ),
    )

    models = await ollama.list_models()

    assert len(models) == 1
    assert models[0].id == "qwen3:8b"
    assert models[0].name == "qwen3:8b"
    assert models[0].supports_streaming is True


@respx.mock
@pytest.mark.asyncio
async def test_chat_404_raises_model_not_found(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """A 404 from Ollama (unknown model) is mapped to `ModelNotFoundError`, not a raw HTTP error."""
    respx_mock.post("/api/chat").mock(return_value=httpx.Response(404, json={"error": "not found"}))

    with pytest.raises(ModelNotFoundError):
        await ollama.chat(
            ChatRequest(
                model="missing-model",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )


@respx.mock
@pytest.mark.asyncio
async def test_chat_500_raises_provider_unavailable(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """A 5xx from Ollama is mapped to `ProviderUnavailableError`, signaling a transient server issue."""
    respx_mock.post("/api/chat").mock(return_value=httpx.Response(500, text="internal error"))

    with pytest.raises(ProviderUnavailableError):
        await ollama.chat(
            ChatRequest(
                model="qwen3:8b",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )


@respx.mock
@pytest.mark.asyncio
async def test_chat_timeout_raises_provider_unavailable(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """A request timeout (no response received at all) is also mapped to `ProviderUnavailableError`."""
    respx_mock.post("/api/chat").mock(side_effect=httpx.TimeoutException("timed out"))

    with pytest.raises(ProviderUnavailableError):
        await ollama.chat(
            ChatRequest(
                model="qwen3:8b",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )


@respx.mock
@pytest.mark.asyncio
async def test_chat_connection_failure_raises_provider_unavailable(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """A connection-level failure (e.g. server not running) is mapped to `ProviderUnavailableError`."""
    respx_mock.post("/api/chat").mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(ProviderUnavailableError):
        await ollama.chat(
            ChatRequest(
                model="qwen3:8b",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )


@pytest.mark.asyncio
async def test_close_closes_owned_client() -> None:
    """`close()` closes the HTTP client when the provider created it itself (no external client passed)."""
    provider = OllamaProvider(base_url=BASE_URL)
    provider._client.aclose = AsyncMock()  # noqa: SLF001

    await provider.close()

    provider._client.aclose.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_close_does_not_close_external_client() -> None:
    """`close()` must NOT close an `httpx.AsyncClient` supplied by the caller, since they own its lifecycle."""
    external_client = httpx.AsyncClient(base_url=BASE_URL)
    external_client.aclose = AsyncMock()
    provider = OllamaProvider(base_url=BASE_URL, http_client=external_client)

    await provider.close()

    external_client.aclose.assert_not_awaited()
    await external_client.aclose()


@pytest.mark.asyncio
async def test_async_context_manager_closes_owned_client() -> None:
    """Using `OllamaProvider` as an async context manager closes its owned client on exit."""
    async with OllamaProvider(base_url=BASE_URL) as provider:
        provider._client.aclose = AsyncMock()  # noqa: SLF001
        assert isinstance(provider, OllamaProvider)

    provider._client.aclose.assert_awaited_once()  # noqa: SLF001


@respx.mock
@pytest.mark.asyncio
async def test_chat_does_not_leak_httpx_exceptions(
    ollama: OllamaProvider,
    respx_mock: respx.MockRouter,
) -> None:
    """
    The raised error is a `ProviderError`, never a raw `httpx.HTTPError` subclass.

    This is an abstraction-boundary check: callers of `BaseProvider.chat()`
    should only ever need to catch the gateway's own exception hierarchy,
    never `httpx`-specific types. `isinstance(exc, httpx.HTTPError)` being
    `False` catches a regression where a raw httpx exception happened to
    also be a `ProviderError` subclass (or wasn't wrapped at all).
    """
    respx_mock.post("/api/chat").mock(return_value=httpx.Response(502, text="bad gateway"))

    with pytest.raises(ProviderError) as exc_info:
        await ollama.chat(
            ChatRequest(
                model="qwen3:8b",
                messages=[ChatMessage(role="user", content="Hi")],
            ),
        )

    assert not isinstance(exc_info.value, httpx.HTTPError)
