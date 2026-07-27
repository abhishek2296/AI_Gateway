"""Anthropic Messages API adapter implementing :class:`~src.providers.base.BaseProvider`."""

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
from src.providers.exceptions import UnsupportedCapabilityError
from src.providers.http_auth import anthropic_headers
from src.providers.http_errors import HTTPErrorMapper
from src.providers.http_mixin import HTTPProviderMixin, require_non_empty_string
from src.providers.registry import register_provider
from src.providers.retry import retry_async
from src.providers.streaming import iter_sse_events
from src.providers.validation import ModelListCache

logger = logging.getLogger(__name__)

_PROVIDER = "anthropic"
_DEFAULT_BASE_URL = "https://api.anthropic.com"


class AnthropicProvider(HTTPProviderMixin, BaseProvider):
    """HTTP adapter for the Anthropic Messages API."""

    provider_name = _PROVIDER

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        api_version: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = require_non_empty_string(api_key, "api_key")
        version = api_version or settings.ANTHROPIC_API_VERSION
        self._headers = anthropic_headers(self._api_key, api_version=version)
        self._mapper = HTTPErrorMapper(_PROVIDER, unavailable_message="Anthropic is unavailable.")
        self._max_retries = (
            max_retries if max_retries is not None else settings.PROVIDER_HTTP_MAX_RETRIES
        )
        self._model_cache = ModelListCache(settings.PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS)
        self._init_http_client(
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            provider_label="AnthropicProvider",
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = _build_messages_payload(request)
        started = time.perf_counter()
        logger.info("Anthropic chat request started (model=%s)", request.model)

        async def _call() -> httpx.Response:
            response = await self._client.post(
                "/v1/messages",
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
            "Anthropic chat request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_chat_response(data)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        payload = _build_messages_payload(request, stream=True)
        started = time.perf_counter()
        logger.info("Anthropic stream request started (model=%s)", request.model)

        try:
            async with self._client.stream(
                "POST",
                "/v1/messages",
                json=payload,
                headers=self._headers,
            ) as response:
                response.raise_for_status()
                async for event_type, data in iter_sse_events(response):
                    chunk = _map_stream_event(event_type, data)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPStatusError as exc:
            raise self._mapper.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise self._mapper.map_request_error(exc) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Anthropic stream request finished (model=%s, latency_ms=%.2f)",
                request.model,
                elapsed_ms,
            )

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        raise UnsupportedCapabilityError(
            "Anthropic does not expose an embeddings API.",
            provider=_PROVIDER,
        )

    async def list_models(self) -> Sequence[ModelInfo]:
        cached = self._model_cache.get(_PROVIDER)
        if cached is not None:
            return cached

        async def _call() -> httpx.Response:
            response = await self._client.get("/v1/models", headers=self._headers)
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
        logger.info("Anthropic health_check started")
        try:
            response = await self._client.get("/v1/models", headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("Anthropic health_check failed: %s", type(exc).__name__)
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message="Unable to reach Anthropic.",
                details={"error": type(exc).__name__},
            )
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message=f"Anthropic returned HTTP {exc.response.status_code}.",
                details={"status_code": str(exc.response.status_code)},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_models = data.get("data", [])
        count = len(raw_models) if isinstance(raw_models, list) else None
        return HealthCheckResult(
            healthy=True,
            latency_ms=elapsed_ms,
            message="Anthropic is reachable.",
            models_available=count,
        )


def _build_messages_payload(request: ChatRequest, *, stream: bool = False) -> dict[str, Any]:
    system_prompt, messages = _split_system_messages(request.messages)
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [_map_message(message) for message in messages],
        "max_tokens": request.max_tokens or 1024,
        "stream": stream,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop_sequences"] = list(request.stop)
    if request.tools:
        payload["tools"] = [_map_tool(tool) for tool in request.tools]
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    if request.response_format is not None and request.response_format.type == "json":
        payload["response_format"] = {"type": "json_object"}
    return payload


def _split_system_messages(
    messages: Sequence[ChatMessage],
) -> tuple[str | None, list[ChatMessage]]:
    system_parts: list[str] = []
    remaining: list[ChatMessage] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
            continue
        remaining.append(message)
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, remaining


def _map_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool" or message.tool_call_id:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                },
            ],
        }
    if message.tool_calls:
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": _parse_json_object(call.arguments),
                },
            )
        return {"role": message.role, "content": content}

    content_blocks = _map_message_content(message)
    return {"role": message.role, "content": content_blocks}


def _map_message_content(message: ChatMessage) -> list[dict[str, Any]] | str:
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
    if isinstance(part, ImagePart) and part.base64_data:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": part.base64_data,
                },
            },
        ]
    if isinstance(part, ImagePart) and part.url:
        return [{"type": "text", "text": f"[image:{part.url}]"}]
    return []


def _map_tool(tool: ToolDefinition) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "name": tool.name,
        "input_schema": dict(tool.parameters or {"type": "object", "properties": {}}),
    }
    if tool.description:
        definition["description"] = tool.description
    return definition


def _map_chat_response(data: Mapping[str, Any]) -> ChatResponse:
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_parts: list[str] = []
    for block in data.get("content") or []:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            content_parts.append(str(block.get("text") or ""))
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=_serialize_json(block.get("input")),
                ),
            )
        elif block_type == "thinking":
            thinking_parts.append(str(block.get("thinking") or ""))

    details: dict[str, str] | None = None
    if thinking_parts:
        details = {"thinking": "\n".join(thinking_parts)}

    return ChatResponse(
        content="".join(content_parts),
        model=str(data.get("model", "")),
        finish_reason=_optional_str(data.get("stop_reason")),
        usage=_map_usage(data.get("usage")),
        provider_response_id=_optional_str(data.get("id")),
        tool_calls=tuple(tool_calls) if tool_calls else None,
        details=details,
    )


def _map_stream_event(
    event_type: str | None,
    data: Mapping[str, Any],
) -> ChatStreamChunk | None:
    if event_type == "content_block_delta":
        delta = data.get("delta") or {}
        if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
            return ChatStreamChunk(content=str(delta.get("text") or ""))
        if isinstance(delta, Mapping) and delta.get("type") == "thinking_delta":
            return ChatStreamChunk(
                content="",
                details={"thinking_delta": str(delta.get("thinking") or "")},
            )
    if event_type == "message_delta":
        delta = data.get("delta") or {}
        usage = _map_usage(data.get("usage"))
        finish_reason = None
        if isinstance(delta, Mapping):
            finish_reason = _optional_str(delta.get("stop_reason"))
        return ChatStreamChunk(content="", finish_reason=finish_reason, usage=usage)
    return None


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
                display_name=str(item.get("display_name") or model_id),
                supports_streaming=True,
                supports_embeddings=False,
                supports_vision=True,
                supports_tools=True,
                supports_json=True,
            ),
        )
    return tuple(models)


def _map_usage(raw_usage: object) -> TokenUsage | None:
    if not isinstance(raw_usage, Mapping):
        return None
    prompt = _as_int(raw_usage.get("input_tokens"))
    completion = _as_int(raw_usage.get("output_tokens"))
    if prompt is None and completion is None:
        return None
    total = (prompt or 0) + (completion or 0)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        import json

        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _serialize_json(value: object) -> str:
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value)


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


register_provider(AnthropicProvider)
