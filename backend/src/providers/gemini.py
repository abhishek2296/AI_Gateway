"""Google Gemini API adapter implementing :class:`~src.providers.base.BaseProvider`."""

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
from src.providers.http_auth import gemini_headers
from src.providers.http_errors import HTTPErrorMapper
from src.providers.http_mixin import HTTPProviderMixin, require_non_empty_string
from src.providers.registry import register_provider
from src.providers.retry import retry_async
from src.providers.streaming import iter_ndjson, iter_sse_json
from src.providers.validation import ModelListCache

logger = logging.getLogger(__name__)

_PROVIDER = "gemini"
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiProvider(HTTPProviderMixin, BaseProvider):
    """HTTP adapter for the Google Gemini Generate Content API."""

    provider_name = _PROVIDER

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = require_non_empty_string(api_key, "api_key")
        self._headers = gemini_headers(self._api_key)
        self._mapper = HTTPErrorMapper(_PROVIDER, unavailable_message="Gemini is unavailable.")
        self._max_retries = (
            max_retries if max_retries is not None else settings.PROVIDER_HTTP_MAX_RETRIES
        )
        self._model_cache = ModelListCache(settings.PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS)
        self._init_http_client(
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            provider_label="GeminiProvider",
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model_path = _model_path(request.model)
        payload = _build_generate_payload(request)
        started = time.perf_counter()
        logger.info("Gemini chat request started (model=%s)", request.model)

        async def _call() -> httpx.Response:
            response = await self._client.post(
                f"/v1beta/{model_path}:generateContent",
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
            "Gemini chat request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_chat_response(data, model=request.model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        model_path = _model_path(request.model)
        payload = _build_generate_payload(request)
        started = time.perf_counter()
        logger.info("Gemini stream request started (model=%s)", request.model)

        try:
            async with self._client.stream(
                "POST",
                f"/v1beta/{model_path}:streamGenerateContent",
                json=payload,
                headers=self._headers,
                params={"alt": "sse"},
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    async for data in iter_sse_json(response):
                        chunk = _map_stream_chunk(data)
                        if chunk is not None:
                            yield chunk
                else:
                    async for data in iter_ndjson(response):
                        chunk = _map_stream_chunk(data)
                        if chunk is not None:
                            yield chunk
        except httpx.HTTPStatusError as exc:
            raise self._mapper.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise self._mapper.map_request_error(exc) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Gemini stream request finished (model=%s, latency_ms=%.2f)",
                request.model,
                elapsed_ms,
            )

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        model_path = _model_path(request.model)
        inputs = request.input if isinstance(request.input, list) else [request.input]
        payload = {
            "content": {"parts": [{"text": text} for text in inputs]},
        }
        started = time.perf_counter()
        logger.info("Gemini embeddings request started (model=%s)", request.model)

        async def _call() -> httpx.Response:
            response = await self._client.post(
                f"/v1beta/{model_path}:embedContent",
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
            "Gemini embeddings request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_embeddings_response(data, model=request.model)

    async def list_models(self) -> Sequence[ModelInfo]:
        cached = self._model_cache.get(_PROVIDER)
        if cached is not None:
            return cached

        async def _call() -> httpx.Response:
            response = await self._client.get("/v1beta/models", headers=self._headers)
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
        normalized = _model_path(model)
        short = normalized.removeprefix("models/")
        return any(
            item.id == model
            or item.name == model
            or item.id == normalized
            or item.name == short
            for item in cached
        )

    async def health_check(self) -> HealthCheckResult:
        started = time.perf_counter()
        logger.info("Gemini health_check started")
        try:
            response = await self._client.get("/v1beta/models", headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("Gemini health_check failed: %s", type(exc).__name__)
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message="Unable to reach Gemini.",
                details={"error": type(exc).__name__},
            )
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message=f"Gemini returned HTTP {exc.response.status_code}.",
                details={"status_code": str(exc.response.status_code)},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_models = data.get("models", [])
        count = len(raw_models) if isinstance(raw_models, list) else None
        return HealthCheckResult(
            healthy=True,
            latency_ms=elapsed_ms,
            message="Gemini is reachable.",
            models_available=count,
        )


def _model_path(model: str) -> str:
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _build_generate_payload(request: ChatRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contents": [_map_message(message) for message in request.messages if message.role != "system"],
    }
    system_messages = [message.content for message in request.messages if message.role == "system"]
    if system_messages:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_messages)}],
        }
    generation_config: dict[str, Any] = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.max_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_tokens
    if request.top_p is not None:
        generation_config["topP"] = request.top_p
    if request.stop:
        generation_config["stopSequences"] = list(request.stop)
    if request.response_format is not None:
        generation_config.update(_map_response_format(request.response_format))
    if generation_config:
        payload["generationConfig"] = generation_config
    if request.tools:
        payload["tools"] = [{"functionDeclarations": [_map_tool(tool) for tool in request.tools]}]
    safety = (
        request.provider_options.get("safetySettings")
        if request.provider_options is not None
        else None
    )
    if isinstance(safety, list):
        payload["safetySettings"] = safety
    return payload


def _map_message(message: ChatMessage) -> dict[str, Any]:
    role = "model" if message.role == "assistant" else "user"
    parts: list[dict[str, Any]] = []
    if message.tool_call_id:
        parts.append(
            {
                "functionResponse": {
                    "name": message.tool_call_id,
                    "response": {"output": message.content},
                },
            },
        )
        return {"role": role, "parts": parts}
    if message.tool_calls:
        for call in message.tool_calls:
            parts.append(
                {
                    "functionCall": {
                        "name": call.name,
                        "args": _parse_json_object(call.arguments),
                    },
                },
            )
        if message.content:
            parts.insert(0, {"text": message.content})
        return {"role": role, "parts": parts}
    if message.parts:
        for part in message.parts:
            parts.extend(_map_content_part(part))
        if message.content:
            parts.insert(0, {"text": message.content})
    elif message.content:
        parts.append({"text": message.content})
    return {"role": role, "parts": parts}


def _map_content_part(part: ContentPart) -> list[dict[str, Any]]:
    if isinstance(part, TextPart):
        return [{"text": part.text}]
    if isinstance(part, ImagePart) and part.base64_data:
        return [
            {
                "inlineData": {
                    "mimeType": part.media_type,
                    "data": part.base64_data,
                },
            },
        ]
    if isinstance(part, ImagePart) and part.url:
        return [{"text": f"[image:{part.url}]"}]
    return []


def _map_tool(tool: ToolDefinition) -> dict[str, Any]:
    declaration: dict[str, Any] = {"name": tool.name}
    if tool.description:
        declaration["description"] = tool.description
    if tool.parameters is not None:
        declaration["parameters"] = dict(tool.parameters)
    return declaration


def _map_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    if response_format.type == "json_schema" and response_format.json_schema is not None:
        schema = response_format.json_schema.get("schema", response_format.json_schema)
        return {"responseMimeType": "application/json", "responseSchema": schema}
    if response_format.type == "json":
        return {"responseMimeType": "application/json"}
    return {}


def _map_chat_response(data: Mapping[str, Any], *, model: str) -> ChatResponse:
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        if "text" in part:
            text_parts.append(str(part.get("text") or ""))
        function_call = part.get("functionCall")
        if isinstance(function_call, Mapping):
            tool_calls.append(
                ToolCall(
                    id=str(function_call.get("name") or ""),
                    name=str(function_call.get("name") or ""),
                    arguments=_serialize_json(function_call.get("args")),
                ),
            )
    finish_reason = None
    if isinstance(candidate, Mapping):
        finish_reason = _optional_str(candidate.get("finishReason"))
    return ChatResponse(
        content="".join(text_parts),
        model=model,
        finish_reason=finish_reason,
        usage=_map_usage(data.get("usageMetadata")),
        tool_calls=tuple(tool_calls) if tool_calls else None,
    )


def _map_stream_chunk(data: Mapping[str, Any]) -> ChatStreamChunk | None:
    response = _map_chat_response(data, model="")
    if not response.content and response.finish_reason is None and not response.tool_calls:
        return None
    return ChatStreamChunk(
        content=response.content,
        finish_reason=response.finish_reason,
        usage=response.usage,
        tool_calls=response.tool_calls,
    )


def _map_embeddings_response(data: Mapping[str, Any], *, model: str) -> EmbeddingsResponse:
    embedding = data.get("embedding") or {}
    values = embedding.get("values") if isinstance(embedding, Mapping) else None
    vectors: list[list[float]] = []
    if isinstance(values, list):
        vectors.append([float(value) for value in values])
    return EmbeddingsResponse(
        model=model,
        embeddings=vectors,
        usage=_map_usage(data.get("usageMetadata")),
    )


def _map_models(data: Mapping[str, Any]) -> tuple[ModelInfo, ...]:
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        return ()
    models: list[ModelInfo] = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        model_name = str(item.get("name") or "")
        if not model_name:
            continue
        short_name = model_name.removeprefix("models/")
        supports_embed = "embed" in short_name.lower()
        models.append(
            ModelInfo(
                id=model_name,
                name=short_name,
                display_name=short_name,
                supports_streaming=True,
                supports_embeddings=supports_embed,
                supports_vision=True,
                supports_tools=True,
                supports_json=True,
            ),
        )
    return tuple(models)


def _map_usage(raw_usage: object) -> TokenUsage | None:
    if not isinstance(raw_usage, Mapping):
        return None
    prompt = _as_int(raw_usage.get("promptTokenCount"))
    completion = _as_int(raw_usage.get("candidatesTokenCount"))
    total = _as_int(raw_usage.get("totalTokenCount"))
    if prompt is None and completion is None and total is None:
        return None
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


register_provider(GeminiProvider)
