"""OpenAI API adapter implementing :class:`~src.providers.base.BaseProvider`."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from src.core.config import get_settings
from src.providers.base import (
    BaseProvider,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ContentPart,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthCheckResult,
    ImagePart,
    ModelInfo,
    ResponseFormat,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from src.providers.exceptions import ProviderError
from src.providers.http_auth import openai_headers
from src.providers.http_errors import HTTPErrorMapper
from src.providers.http_mixin import HTTPProviderMixin, require_non_empty_string
from src.providers.registry import register_provider
from src.providers.retry import retry_async
from src.providers.streaming import iter_sse_json
from src.providers.validation import ModelListCache

logger = logging.getLogger(__name__)

_PROVIDER = "openai"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(HTTPProviderMixin, BaseProvider):
    """HTTP adapter for the OpenAI Chat Completions and Embeddings APIs."""

    provider_name = _PROVIDER

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        organization: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = require_non_empty_string(api_key, "api_key")
        self._organization = organization.strip() if organization else None
        self._headers = openai_headers(self._api_key, organization=self._organization)
        self._mapper = HTTPErrorMapper(_PROVIDER, unavailable_message="OpenAI is unavailable.")
        self._max_retries = (
            max_retries if max_retries is not None else settings.PROVIDER_HTTP_MAX_RETRIES
        )
        self._model_cache = ModelListCache(settings.PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS)
        self._init_http_client(
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            provider_label="OpenAIProvider",
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = _build_chat_payload(request, stream=False)
        started = time.perf_counter()
        logger.info("OpenAI chat request started (model=%s)", request.model)

        async def _call() -> httpx.Response:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()
            return response

        try:
            response = await retry_async(
                _call,
                mapper=self._mapper,
                max_retries=self._max_retries,
            )
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._mapper.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise self._mapper.map_request_error(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "OpenAI chat request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_chat_response(data)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        payload = _build_chat_payload(request, stream=True)
        started = time.perf_counter()
        logger.info("OpenAI stream request started (model=%s)", request.model)

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers=self._headers,
            ) as response:
                response.raise_for_status()
                async for data in iter_sse_json(response):
                    yield _map_stream_chunk(data)
        except httpx.HTTPStatusError as exc:
            raise self._mapper.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise self._mapper.map_request_error(exc) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "OpenAI stream request finished (model=%s, latency_ms=%.2f)",
                request.model,
                elapsed_ms,
            )

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        payload = {"model": request.model, "input": request.input}
        started = time.perf_counter()
        logger.info("OpenAI embeddings request started (model=%s)", request.model)

        async def _call() -> httpx.Response:
            response = await self._client.post(
                "/embeddings",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()
            return response

        try:
            response = await retry_async(_call, mapper=self._mapper, max_retries=self._max_retries)
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._mapper.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise self._mapper.map_request_error(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "OpenAI embeddings request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_embeddings_response(data)

    async def list_models(self) -> Sequence[ModelInfo]:
        cached = self._model_cache.get(_PROVIDER)
        if cached is not None:
            return cached

        async def _call() -> httpx.Response:
            response = await self._client.get("/models", headers=self._headers)
            response.raise_for_status()
            return response

        try:
            response = await retry_async(_call, mapper=self._mapper, max_retries=self._max_retries)
            models = _map_models(response.json())
        except httpx.HTTPStatusError as exc:
            raise self._mapper.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise self._mapper.map_request_error(exc) from exc

        self._model_cache.set(_PROVIDER, models)
        return models

    async def validate_model(self, model: str) -> bool:
        cached = self._model_cache.get(_PROVIDER)
        if cached is None:
            cached = await self.list_models()
        return any(item.id == model or item.name == model for item in cached)

    async def health_check(self) -> HealthCheckResult:
        started = time.perf_counter()
        logger.info("OpenAI health_check started")
        try:
            response = await self._client.get("/models", headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("OpenAI health_check failed: %s", type(exc).__name__)
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message="Unable to reach OpenAI.",
                details={"error": type(exc).__name__},
            )
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message=f"OpenAI returned HTTP {exc.response.status_code}.",
                details={"status_code": str(exc.response.status_code)},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_models = data.get("data", [])
        count = len(raw_models) if isinstance(raw_models, list) else None
        return HealthCheckResult(
            healthy=True,
            latency_ms=elapsed_ms,
            message="OpenAI is reachable.",
            models_available=count,
        )


def _build_chat_payload(request: ChatRequest, *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [_map_message(message) for message in request.messages],
        "stream": stream,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop"] = list(request.stop)
    if request.tools:
        payload["tools"] = [_map_tool(tool) for tool in request.tools]
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        payload["response_format"] = _map_response_format(request.response_format)
    return payload


def _map_message(message: ChatMessage) -> dict[str, Any]:
    if message.tool_call_id:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if message.tool_calls:
        return {
            "role": message.role,
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ],
        }
    content = _map_message_content(message)
    return {"role": message.role, "content": content}


def _map_message_content(message: ChatMessage) -> str | list[dict[str, Any]]:
    if message.parts:
        blocks: list[dict[str, Any]] = []
        for part in message.parts:
            blocks.extend(_map_content_part(part))
        if message.content:
            blocks.insert(0, {"type": "text", "text": message.content})
        return blocks
    return message.content


def _map_content_part(part: ContentPart) -> list[dict[str, Any]]:
    if isinstance(part, TextPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, ImagePart):
        if part.url:
            return [{"type": "image_url", "image_url": {"url": part.url}}]
        if part.base64_data:
            data_url = f"data:{part.media_type};base64,{part.base64_data}"
            return [{"type": "image_url", "image_url": {"url": data_url}}]
    return []


def _map_tool(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.name}
    if tool.description:
        function["description"] = tool.description
    if tool.parameters is not None:
        function["parameters"] = dict(tool.parameters)
    return {"type": "function", "function": function}


def _map_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    if response_format.type == "json_schema" and response_format.json_schema is not None:
        return {
            "type": "json_schema",
            "json_schema": dict(response_format.json_schema),
        }
    if response_format.type == "json":
        return {"type": "json_object"}
    return {"type": "text"}


def _map_chat_response(data: Mapping[str, Any]) -> ChatResponse:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    tool_calls = _map_tool_calls(message.get("tool_calls"))
    return ChatResponse(
        content=str(message.get("content") or ""),
        model=str(data.get("model", "")),
        finish_reason=_optional_str(choice.get("finish_reason")),
        usage=_map_usage(data.get("usage")),
        provider_response_id=_optional_str(data.get("id")),
        tool_calls=tool_calls,
    )


def _map_stream_chunk(data: Mapping[str, Any]) -> ChatStreamChunk:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    tool_calls = _map_tool_calls(delta.get("tool_calls"))
    return ChatStreamChunk(
        content=str(delta.get("content") or ""),
        finish_reason=_optional_str(choice.get("finish_reason")),
        usage=_map_usage(data.get("usage")) if data.get("usage") else None,
        tool_calls=tool_calls,
    )


def _map_tool_calls(raw_calls: object) -> tuple[ToolCall, ...] | None:
    if not isinstance(raw_calls, list) or not raw_calls:
        return None
    calls: list[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") or {}
        if not isinstance(function, Mapping):
            continue
        call_id = str(item.get("id") or "")
        name = str(function.get("name") or "")
        arguments = str(function.get("arguments") or "")
        if call_id and name:
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return tuple(calls) if calls else None


def _map_embeddings_response(data: Mapping[str, Any]) -> EmbeddingsResponse:
    raw_data = data.get("data", [])
    embeddings: list[list[float]] = []
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, Mapping):
                vector = item.get("embedding")
                if isinstance(vector, list):
                    embeddings.append([float(value) for value in vector])
    return EmbeddingsResponse(
        model=str(data.get("model", "")),
        embeddings=embeddings,
        usage=_map_usage(data.get("usage")),
    )


def _map_models(data: Mapping[str, Any]) -> tuple[ModelInfo, ...]:
    raw_models = data.get("data", [])
    if not isinstance(raw_models, list):
        return ()
    models: list[ModelInfo] = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id") or "")
        if not model_id:
            continue
        models.append(
            ModelInfo(
                id=model_id,
                name=model_id,
                display_name=model_id,
                supports_streaming=True,
                supports_embeddings=model_id.startswith("text-embedding"),
                supports_vision="vision" in model_id or model_id.startswith("gpt-4"),
                supports_tools=model_id.startswith("gpt-"),
                supports_json=model_id.startswith("gpt-"),
            ),
        )
    return tuple(models)


def _map_usage(raw_usage: object) -> TokenUsage | None:
    if not isinstance(raw_usage, Mapping):
        return None
    prompt = _as_int(raw_usage.get("prompt_tokens"))
    completion = _as_int(raw_usage.get("completion_tokens"))
    total = _as_int(raw_usage.get("total_tokens"))
    if prompt is None and completion is None and total is None:
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


register_provider(OpenAIProvider)
