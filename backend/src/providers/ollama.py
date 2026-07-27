"""Ollama REST adapter implementing :class:`~src.providers.base.BaseProvider`."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from src.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthCheckResult,
    ModelInfo,
    TokenUsage,
)
from src.providers.exceptions import ProviderError
from src.providers.http_errors import HTTPErrorMapper
from src.providers.registry import register_provider
from src.providers.http_mixin import HTTPProviderMixin

logger = logging.getLogger(__name__)

_PROVIDER = "ollama"
_ERROR_MAPPER = HTTPErrorMapper(_PROVIDER, unavailable_message="Ollama is unavailable.")


class OllamaProvider(HTTPProviderMixin, BaseProvider):
    """
    HTTP adapter for the Ollama REST API.

    Owns transport concerns only: request construction, response parsing, error
    translation, and client lifecycle. Business rules remain in the service layer.
    """

    provider_name = _PROVIDER

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._init_http_client(
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            provider_label="OllamaProvider",
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming chat completion via ``POST /api/chat``."""
        payload = _build_chat_payload(request, stream=False)
        started = time.perf_counter()
        logger.info("Ollama chat request started (model=%s)", request.model)

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise _ERROR_MAPPER.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise _ERROR_MAPPER.map_request_error(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Ollama chat request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_chat_response(data)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks via ``POST /api/chat`` with ``stream=true``."""
        payload = _build_chat_payload(request, stream=True)
        started = time.perf_counter()
        logger.info("Ollama stream request started (model=%s)", request.model)

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(
                            "Ollama returned invalid streaming JSON.",
                            provider=_PROVIDER,
                        ) from exc
                    yield _map_stream_chunk(data)
        except httpx.HTTPStatusError as exc:
            raise _ERROR_MAPPER.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise _ERROR_MAPPER.map_request_error(exc) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Ollama stream request finished (model=%s, latency_ms=%.2f)",
                request.model,
                elapsed_ms,
            )

    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Generate embeddings via ``POST /api/embeddings``."""
        payload = {"model": request.model, "input": request.input}
        started = time.perf_counter()
        logger.info("Ollama embeddings request started (model=%s)", request.model)

        try:
            response = await self._client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise _ERROR_MAPPER.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise _ERROR_MAPPER.map_request_error(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Ollama embeddings request completed (model=%s, latency_ms=%.2f)",
            request.model,
            elapsed_ms,
        )
        return _map_embeddings_response(data)

    async def list_models(self) -> Sequence[ModelInfo]:
        """List available models via ``GET /api/tags``."""
        started = time.perf_counter()
        logger.info("Ollama list_models request started")

        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise _ERROR_MAPPER.map_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise _ERROR_MAPPER.map_request_error(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        models = _map_models(data)
        logger.info(
            "Ollama list_models request completed (count=%d, latency_ms=%.2f)",
            len(models),
            elapsed_ms,
        )
        return models

    async def health_check(self) -> HealthCheckResult:
        """
        Probe Ollama readiness via ``GET /api/tags``.

        Connection and HTTP failures are encoded as unhealthy results rather
        than raised exceptions.
        """
        started = time.perf_counter()
        logger.info("Ollama health_check started")

        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("Ollama health_check failed: %s", type(exc).__name__)
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message="Unable to reach Ollama.",
                details={"error": type(exc).__name__},
            )
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "Ollama health_check returned HTTP %d",
                exc.response.status_code,
            )
            return HealthCheckResult(
                healthy=False,
                latency_ms=elapsed_ms,
                message=f"Ollama returned HTTP {exc.response.status_code}.",
                details={"status_code": str(exc.response.status_code)},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        models = data.get("models", [])
        logger.info(
            "Ollama health_check completed (healthy=true, latency_ms=%.2f)",
            elapsed_ms,
        )
        return HealthCheckResult(
            healthy=True,
            latency_ms=elapsed_ms,
            message="Ollama is reachable.",
            models_available=len(models) if isinstance(models, list) else None,
        )


def _build_chat_payload(request: ChatRequest, *, stream: bool) -> dict[str, Any]:
    messages = [{"role": message.role, "content": message.content} for message in request.messages]
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": stream,
    }
    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if options:
        payload["options"] = options
    return payload


def _map_token_usage(data: Mapping[str, Any]) -> TokenUsage | None:
    prompt_tokens = _as_int(data.get("prompt_eval_count"))
    completion_tokens = _as_int(data.get("eval_count"))
    if prompt_tokens is None and completion_tokens is None:
        return None
    total_tokens: int | None = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _map_chat_response(data: Mapping[str, Any]) -> ChatResponse:
    message = data.get("message") or {}
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    return ChatResponse(
        content=str(content),
        model=str(data.get("model", "")),
        finish_reason=_optional_str(data.get("done_reason")),
        usage=_map_token_usage(data),
    )


def _map_stream_chunk(data: Mapping[str, Any]) -> ChatStreamChunk:
    message = data.get("message") or {}
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    done = bool(data.get("done"))
    return ChatStreamChunk(
        content=str(content),
        finish_reason=_optional_str(data.get("done_reason")) if done else None,
        usage=_map_token_usage(data) if done else None,
    )


def _map_embeddings_response(data: Mapping[str, Any]) -> EmbeddingsResponse:
    raw_embeddings = data.get("embeddings", [])
    embeddings: list[list[float]] = []
    if isinstance(raw_embeddings, list):
        for item in raw_embeddings:
            if isinstance(item, list):
                embeddings.append([float(value) for value in item])
    return EmbeddingsResponse(
        model=str(data.get("model", "")),
        embeddings=embeddings,
        usage=_map_token_usage(data),
    )


def _map_models(data: Mapping[str, Any]) -> tuple[ModelInfo, ...]:
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        return ()
    models: list[ModelInfo] = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("model") or "")
        model_id = str(item.get("model") or name)
        if not name:
            continue
        models.append(
            ModelInfo(
                id=model_id,
                name=name,
                display_name=name,
                supports_streaming=True,
                supports_embeddings="embed" in name.lower(),
                supports_vision=True,
                supports_tools=True,
                supports_json=False,
            ),
        )
    return tuple(models)


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


register_provider(OllamaProvider)
