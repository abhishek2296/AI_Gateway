"""Cross-provider contract tests with mocked HTTP."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx

from src.providers.base import ChatMessage, ChatRequest, ChatResponse, HealthCheckResult
from src.providers.anthropic import AnthropicProvider
from src.providers.gemini import GeminiProvider
from src.providers.ollama import OllamaProvider
from src.providers.openai import OpenAIProvider

ProviderFactory = Callable[[respx.MockRouter], Any]


def _ollama_factory(respx_mock: respx.MockRouter) -> OllamaProvider:
    base = "http://ollama.test"
    respx_mock.base_url = base
    return OllamaProvider(base_url=base)


def _openai_factory(respx_mock: respx.MockRouter) -> OpenAIProvider:
    base = "https://api.openai.test"
    respx_mock.base_url = base
    return OpenAIProvider(api_key="test-key", base_url=f"{base}/v1")


def _anthropic_factory(respx_mock: respx.MockRouter) -> AnthropicProvider:
    base = "https://api.anthropic.test"
    respx_mock.base_url = base
    return AnthropicProvider(api_key="test-key", base_url=base)


def _gemini_factory(respx_mock: respx.MockRouter) -> GeminiProvider:
    base = "https://generativelanguage.googleapis.com"
    respx_mock.base_url = base
    return GeminiProvider(api_key="test-key", base_url=base)


PROVIDER_CASES: list[tuple[str, ProviderFactory, Callable[[respx.MockRouter], None]]] = [
    (
        "ollama",
        _ollama_factory,
        lambda mock: mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                    "done_reason": "stop",
                },
            ),
        ),
    ),
    (
        "openai",
        _openai_factory,
        lambda mock: mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gpt-4o",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                },
            ),
        ),
    ),
    (
        "anthropic",
        _anthropic_factory,
        lambda mock: mock.post("/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                },
            ),
        ),
    ),
    (
        "gemini",
        _gemini_factory,
        lambda mock: mock.post("/v1beta/models/gemini-pro:generateContent").mock(
            return_value=httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
            ),
        ),
    ),
]


HEALTH_CASES: list[tuple[str, ProviderFactory, Callable[[respx.MockRouter], None]]] = [
    (
        "ollama",
        _ollama_factory,
        lambda mock: mock.get("/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]}),
        ),
    ),
    (
        "openai",
        _openai_factory,
        lambda mock: mock.get("/v1/models").mock(return_value=httpx.Response(200, json={"data": []})),
    ),
    (
        "anthropic",
        _anthropic_factory,
        lambda mock: mock.get("/v1/models").mock(return_value=httpx.Response(200, json={"data": []})),
    ),
    (
        "gemini",
        _gemini_factory,
        lambda mock: mock.get("/v1beta/models").mock(return_value=httpx.Response(200, json={"models": []})),
    ),
]


MODEL_BY_PROVIDER = {
    "ollama": "qwen3:8b",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-pro",
}



@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory", "register_route"), PROVIDER_CASES)
async def test_provider_chat_contract(
    name: str,
    factory: ProviderFactory,
    register_route: Callable[[respx.MockRouter], None],
    respx_mock: respx.MockRouter,
) -> None:
    register_route(respx_mock)
    provider = factory(respx_mock)

    response = await provider.chat(
        ChatRequest(
            model=MODEL_BY_PROVIDER[name],
            messages=[ChatMessage(role="user", content="hello")],
        ),
    )

    assert isinstance(response, ChatResponse)
    assert response.content == "ok"



@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory", "register_route"), HEALTH_CASES)
async def test_provider_health_check_contract(
    name: str,
    factory: ProviderFactory,
    register_route: Callable[[respx.MockRouter], None],
    respx_mock: respx.MockRouter,
) -> None:
    del name
    register_route(respx_mock)
    provider = factory(respx_mock)

    result = await provider.health_check()

    assert isinstance(result, HealthCheckResult)
    assert result.healthy is True
