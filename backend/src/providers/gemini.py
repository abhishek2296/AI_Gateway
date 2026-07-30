"""
Google Gemini API adapter implementing :class:`~src.providers.base.BaseProvider`.

Translates the gateway's normalized request/response DTOs to and from
Google's Generative Language API (Gemini) payloads. Gemini's API has its
own distinct conventions this module accounts for:

- Model identifiers are addressed via a ``models/<name>`` resource path in
  the URL (e.g. ``/v1beta/models/gemini-pro:generateContent``), not passed
  in the request body — see :func:`_model_path`.
- There is no ``"assistant"`` role; Gemini calls it ``"model"`` — see
  :func:`_map_message`.
- System instructions are a dedicated top-level ``systemInstruction`` field
  (conceptually similar to Anthropic's ``system`` field), not a message with
  a ``"system"`` role.
- Streaming can come back as either classic SSE or plain NDJSON depending on
  request parameters, so this adapter picks the right parser at runtime
  based on the response's ``Content-Type`` — see :meth:`GeminiProvider.stream_chat`.
- Function calling uses ``functionCall``/``functionResponse`` parts embedded
  in a message's ``parts`` array, and function declarations are nested under
  a ``functionDeclarations`` list within a ``tools`` array.
"""

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
    """
    HTTP adapter for the Google Gemini Generate Content API.

    Example:
        >>> import asyncio
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> provider = GeminiProvider(api_key="test-key")
        >>> request = ChatRequest(
        ...     model="gemini-pro",
        ...     messages=[ChatMessage(role="user", content="Hello!")],
        ... )
        >>> response = asyncio.run(provider.chat(request))  # doctest: +SKIP
        >>> asyncio.run(provider.close())
    """

    provider_name = _PROVIDER

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Create a Gemini adapter authenticated with the given API key.

        Args:
            api_key: The caller's Gemini API key. Validated as non-empty via
                :func:`~src.providers.http_mixin.require_non_empty_string`.
            base_url: The Gemini API base URL. Defaults to
                ``"https://generativelanguage.googleapis.com"``.
            timeout: Request timeout in seconds for the internally created
                HTTP client, applied only when ``http_client`` is ``None``.
            http_client: An optional pre-built ``httpx.AsyncClient`` to reuse.
            max_retries: Maximum retry attempts for
                :func:`~src.providers.retry.retry_async`. Defaults to
                ``Settings.PROVIDER_HTTP_MAX_RETRIES`` when not explicitly
                provided.

        Raises:
            ValueError: If ``api_key`` is empty or only whitespace.
        """
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
        """
        Execute a non-streaming chat completion via ``POST /v1beta/{model}:generateContent``.

        Args:
            request: The normalized chat request to execute.

        Returns:
            The normalized chat completion response, mapped via
            :func:`_map_chat_response`.

        Raises:
            ModelNotFoundError: HTTP 404.
            InvalidRequestError: HTTP 400.
            AuthenticationError: HTTP 401/403.
            RateLimitError: HTTP 429, after retries are exhausted.
            ProviderUnavailableError: HTTP 5xx after retries are exhausted,
                or a transport-level failure after retries are exhausted.
            ProviderError: Any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import ChatMessage, ChatRequest
            >>> provider = GeminiProvider(api_key="test-key")
            >>> request = ChatRequest(
            ...     model="gemini-pro",
            ...     messages=[ChatMessage(role="user", content="Hi")],
            ... )
            >>> asyncio.run(provider.chat(request))  # doctest: +SKIP
            ChatResponse(content='Hello!', model='gemini-pro', ...)
        """
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
        """
        Stream chat completion chunks via ``POST /v1beta/{model}:streamGenerateContent``.

        Requests SSE mode explicitly via ``params={"alt": "sse"}``, but
        Gemini does not always honor that consistently across all
        configurations/versions of the API — so rather than assuming SSE
        will be used, this method inspects the actual response
        ``Content-Type`` header and picks the matching parser at runtime:
        :func:`~src.providers.streaming.iter_sse_json` for
        ``text/event-stream`` responses, or
        :func:`~src.providers.streaming.iter_ndjson` for anything else
        (plain newline-delimited JSON). This defensive dispatch avoids the
        adapter silently mis-parsing a response that didn't come back in
        the requested format.

        Args:
            request: The normalized chat request to execute.

        Yields:
            :class:`~src.providers.base.ChatStreamChunk` objects, mapped via
            :func:`_map_stream_chunk`. Chunks that map to "nothing useful"
            (see :func:`_map_stream_chunk`'s ``None`` case) are filtered out
            before yielding.

        Raises:
            ModelNotFoundError: HTTP 404.
            InvalidRequestError: HTTP 400.
            AuthenticationError: HTTP 401/403.
            RateLimitError: HTTP 429.
            ProviderUnavailableError: HTTP 5xx, or a transport-level failure
                at any point before or during streaming.
            ProviderError: Any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import ChatMessage, ChatRequest
            >>> async def demo():
            ...     provider = GeminiProvider(api_key="test-key")
            ...     request = ChatRequest(
            ...         model="gemini-pro",
            ...         messages=[ChatMessage(role="user", content="Hi")],
            ...         stream=True,
            ...     )
            ...     async for chunk in provider.stream_chat(request):
            ...         print(chunk.content, end="")
            >>> asyncio.run(demo())  # doctest: +SKIP
        """
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
        """
        Generate embeddings via ``POST /v1beta/{model}:embedContent``.

        Gemini's ``embedContent`` endpoint embeds a single piece of content
        per call (unlike OpenAI's batch-capable ``/embeddings`` endpoint),
        so a plain-string ``request.input`` is wrapped into a one-element
        list before being packed into the ``content.parts`` structure
        Gemini expects.

        Args:
            request: The normalized embeddings request to execute.

        Returns:
            The normalized embeddings response, mapped via
            :func:`_map_embeddings_response`.

        Raises:
            ModelNotFoundError: HTTP 404.
            InvalidRequestError: HTTP 400.
            AuthenticationError: HTTP 401/403.
            RateLimitError: HTTP 429, after retries are exhausted.
            ProviderUnavailableError: HTTP 5xx after retries are exhausted,
                or a transport-level failure after retries are exhausted.
            ProviderError: Any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import EmbeddingsRequest
            >>> provider = GeminiProvider(api_key="test-key")
            >>> request = EmbeddingsRequest(model="text-embedding-004", input="hello")
            >>> asyncio.run(provider.embeddings(request))  # doctest: +SKIP
            EmbeddingsResponse(model='text-embedding-004', embeddings=[[...]], usage=None)
        """
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
        """
        List available models via ``GET /v1beta/models``, using a short-lived cache.

        Returns:
            A tuple of :class:`~src.providers.base.ModelInfo`, either served
            from :attr:`_model_cache` or freshly fetched and mapped via
            :func:`_map_models`.

        Raises:
            AuthenticationError: HTTP 401/403.
            RateLimitError: HTTP 429, after retries are exhausted.
            ProviderUnavailableError: HTTP 5xx after retries are exhausted,
                or a transport-level failure after retries are exhausted.
            ProviderError: Any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> provider = GeminiProvider(api_key="test-key")
            >>> asyncio.run(provider.list_models())  # doctest: +SKIP
            (ModelInfo(id='models/gemini-pro', name='gemini-pro', ...), ...)
        """
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
        """
        Return whether ``model`` is offered by Gemini, using the cached catalog.

        More involved than the equivalent OpenAI/Anthropic overrides because
        Gemini models can legitimately be referenced two ways — a bare name
        (``"gemini-pro"``) or the full resource path
        (``"models/gemini-pro"``) — and :func:`_map_models` stores ``id`` as
        the full path while ``name`` is the bare form. To match a caller's
        input regardless of which form they used, this method also computes
        both the normalized full path and the bare short form of the
        *input* and checks all four combinations.

        Args:
            model: The model identifier to check, in either bare
                (``"gemini-pro"``) or full-path (``"models/gemini-pro"``)
                form.

        Returns:
            ``True`` if ``model`` (in either form) matches any cached/
            fetched model's ``id`` or ``name``; ``False`` otherwise.

        Raises:
            ProviderError: If a cache miss forces a :meth:`list_models` call
                that itself fails.

        Example:
            >>> import asyncio
            >>> provider = GeminiProvider(api_key="test-key")
            >>> asyncio.run(provider.validate_model("gemini-pro"))  # doctest: +SKIP
            True
            >>> asyncio.run(provider.validate_model("models/gemini-pro"))  # doctest: +SKIP
            True
        """
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
        """
        Probe Gemini readiness via ``GET /v1beta/models``.

        Connection and HTTP failures are encoded as unhealthy results rather
        than raised exceptions.

        Returns:
            A :class:`~src.providers.base.HealthCheckResult` describing
            whether Gemini responded successfully, with the reported model
            count when healthy.

        Raises:
            (none) — every failure path is caught and encoded into the
            returned result.

        Example:
            >>> import asyncio
            >>> provider = GeminiProvider(api_key="test-key")
            >>> asyncio.run(provider.health_check())  # doctest: +SKIP
            HealthCheckResult(healthy=True, ...)
        """
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
    """
    Normalize a model identifier into Gemini's ``models/<name>`` resource path form.

    Gemini's REST API addresses models via a resource path segment in the
    URL (e.g. ``/v1beta/models/gemini-pro:generateContent``) rather than a
    field in the request body. Callers may reasonably supply either the
    bare name or the already-prefixed path, so this function is idempotent:
    prefixing an already-prefixed value would produce
    ``"models/models/gemini-pro"``, which is why the existing prefix is
    checked for first.

    Args:
        model: A model identifier, either bare (``"gemini-pro"``) or already
            in resource-path form (``"models/gemini-pro"``).

    Returns:
        The model identifier guaranteed to start with ``"models/"``.

    Example:
        >>> _model_path("gemini-pro")
        'models/gemini-pro'
        >>> _model_path("models/gemini-pro")
        'models/gemini-pro'
    """
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _build_generate_payload(request: ChatRequest) -> dict[str, Any]:
    """
    Build the JSON body for Gemini's ``generateContent``/``streamGenerateContent`` endpoints.

    Several fields are handled differently from the OpenAI/Anthropic
    adapters:

    - System-role messages are filtered out of ``contents`` and instead
      joined into a single ``systemInstruction`` field, matching Gemini's
      convention of treating system prompts as a top-level field rather
      than a conversation message.
    - Sampling parameters (temperature, max tokens, top_p, stop sequences,
      response format) are nested under a single ``generationConfig``
      object rather than sent as top-level fields.
    - ``provider_options["safetySettings"]`` is read from the normalized
      request's vendor-specific escape hatch
      (``ChatRequest.provider_options``) and passed through only when it's
      actually a list, since Gemini's safety settings have no equivalent
      concept in the other supported vendors and therefore no dedicated
      field on :class:`~src.providers.base.ChatRequest`.

    Args:
        request: The normalized chat request to translate.

    Returns:
        A JSON-serializable dict matching Gemini's ``generateContent``
        request schema. ``generationConfig`` and ``tools`` are omitted
        entirely when there's nothing to put in them, rather than sent as
        empty objects/lists.

    Example:
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> request = ChatRequest(
        ...     model="gemini-pro",
        ...     messages=[
        ...         ChatMessage(role="system", content="Be terse."),
        ...         ChatMessage(role="user", content="Hi"),
        ...     ],
        ...     temperature=0.5,
        ... )
        >>> _build_generate_payload(request)
        {'contents': [{'role': 'user', 'parts': [{'text': 'Hi'}]}], 'systemInstruction': {'parts': [{'text': 'Be terse.'}]}, 'generationConfig': {'temperature': 0.5}}
    """
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
    """
    Map a normalized :class:`~src.providers.base.ChatMessage` to Gemini's message shape.

    Gemini has no ``"tool"`` role at all — both a tool result
    (``functionResponse``) and a tool call (``functionCall``) are always
    sent as parts of a ``"user"``/``"model"`` turn, never as a distinct
    role. This function also renames ``"assistant"`` to ``"model"``, since
    that's the only role name Gemini recognizes for the model's own turns;
    every other role (including ``"user"`` and ``"tool"``) maps to
    ``"user"``.

    Args:
        message: The message to translate.

    Returns:
        A dict of the form ``{"role", "parts"}``, where ``role`` is
        ``"model"`` for assistant messages and ``"user"`` otherwise, and
        ``parts`` holds a ``functionResponse``, one or more
        ``functionCall``s, or plain text/multimodal content blocks,
        depending on which fields are populated on ``message``.

    Example:
        >>> _map_message(ChatMessage(role="user", content="Hi"))
        {'role': 'user', 'parts': [{'text': 'Hi'}]}
        >>> _map_message(ChatMessage(role="assistant", content="Hello!"))
        {'role': 'model', 'parts': [{'text': 'Hello!'}]}
    """
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
    """
    Map one :data:`~src.providers.base.ContentPart` to Gemini content part(s).

    Like Anthropic, Gemini's inline image parts only support base64 data
    (``inlineData``), not a remote URL — so a URL-only :class:`ImagePart` is
    degraded to a descriptive text part (``"[image:<url>]"``) rather than
    being dropped silently, matching the same fallback strategy used by
    ``anthropic._map_content_part``.

    Args:
        part: A single text or image content part.

    Returns:
        A list containing the mapped part, or an empty list if ``part`` is
        an :class:`ImagePart` with neither ``base64_data`` nor ``url`` set.

    Example:
        >>> _map_content_part(TextPart(text="hi"))
        [{'text': 'hi'}]
        >>> _map_content_part(ImagePart(base64_data="aGk=", media_type="image/png"))
        [{'inlineData': {'mimeType': 'image/png', 'data': 'aGk='}}]
    """
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
    """
    Map a :class:`~src.providers.base.ToolDefinition` to a Gemini function declaration.

    Args:
        tool: The tool definition to translate.

    Returns:
        A dict of the form ``{"name", ["description"], ["parameters"]}``,
        intended to be wrapped in a ``functionDeclarations`` list (see
        :func:`_build_generate_payload`).

    Example:
        >>> _map_tool(ToolDefinition(name="get_weather", description="Get weather."))
        {'name': 'get_weather', 'description': 'Get weather.'}
    """
    declaration: dict[str, Any] = {"name": tool.name}
    if tool.description:
        declaration["description"] = tool.description
    if tool.parameters is not None:
        declaration["parameters"] = dict(tool.parameters)
    return declaration


def _map_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    """
    Map a :class:`~src.providers.base.ResponseFormat` into Gemini's ``generationConfig`` fields.

    Gemini expresses structured output via ``responseMimeType``/
    ``responseSchema`` fields nested in ``generationConfig``, rather than a
    single dedicated ``response_format``-style field the way OpenAI/
    Anthropic do — this function returns a dict meant to be merged
    (``generation_config.update(...)``) into that larger config object.

    Args:
        response_format: The structured-output configuration to translate.

    Returns:
        ``{"responseMimeType": "application/json", "responseSchema": ...}``
        when a schema is specified (unwrapping a ``{"schema": {...}}``
        envelope if present, to tolerate either a bare schema or a
        JSON-Schema-style wrapped one being passed in);
        ``{"responseMimeType": "application/json"}`` for unconstrained JSON
        mode; or an empty dict for plain text (since Gemini's default
        behavior needs no explicit config).

    Example:
        >>> _map_response_format(ResponseFormat(type="json"))
        {'responseMimeType': 'application/json'}
        >>> _map_response_format(ResponseFormat(type="text"))
        {}
    """
    if response_format.type == "json_schema" and response_format.json_schema is not None:
        schema = response_format.json_schema.get("schema", response_format.json_schema)
        return {"responseMimeType": "application/json", "responseSchema": schema}
    if response_format.type == "json":
        return {"responseMimeType": "application/json"}
    return {}


def _map_chat_response(data: Mapping[str, Any], *, model: str) -> ChatResponse:
    """
    Map a Gemini ``generateContent`` response body into a :class:`ChatResponse`.

    Only the first entry in ``candidates`` is mapped, mirroring how the
    OpenAI adapter only reads ``choices[0]`` — Gemini can also return
    multiple candidates, but this gateway always requests (and expects)
    just one.

    Args:
        data: The parsed JSON response body from Gemini.
        model: The originally-requested model identifier, passed in
            explicitly rather than read from ``data`` because Gemini's
            response body does not echo back which model served the
            request the way OpenAI/Ollama do.

    Returns:
        The normalized :class:`~src.providers.base.ChatResponse`, with
        ``content`` built by concatenating every text part in the response
        (a candidate's ``parts`` array may interleave multiple text
        fragments and function calls).

    Example:
        >>> _map_chat_response(
        ...     {
        ...         "candidates": [
        ...             {
        ...                 "content": {"parts": [{"text": "Hi!"}]},
        ...                 "finishReason": "STOP",
        ...             },
        ...         ],
        ...     },
        ...     model="gemini-pro",
        ... )
        ChatResponse(content='Hi!', model='gemini-pro', finish_reason='STOP', usage=None, provider_response_id=None, tool_calls=None, details={})
    """
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
            # Gemini's functionCall has no separate call-id field, so the
            # function name doubles as the ToolCall id here (see
            # ToolCall.id's docstring for this same accommodation).
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
    """
    Map one Gemini streaming payload into a :class:`ChatStreamChunk`, or ``None`` to skip it.

    Unlike OpenAI/Anthropic (which have a distinct "delta" shape for
    streaming vs. a "full message" shape for non-streaming), each Gemini
    streaming payload is shaped identically to a full non-streaming
    response — so this function simply reuses :func:`_map_chat_response`
    (with an empty ``model``, since it's discarded by the caller for stream
    chunks) rather than duplicating response-parsing logic.

    A chunk that carries no text, no finish reason, and no tool calls is
    considered content-free (e.g. an intermediate bookkeeping payload with
    nothing new to report) and is mapped to ``None`` so the caller
    (:meth:`GeminiProvider.stream_chat`) can filter it out rather than
    yielding an empty, useless chunk to consumers.

    Args:
        data: One parsed streaming payload (from either the SSE or NDJSON
            parser).

    Returns:
        A :class:`~src.providers.base.ChatStreamChunk` if the payload
        carries any content, finish reason, or tool calls; ``None`` otherwise.

    Example:
        >>> _map_stream_chunk({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]})
        ChatStreamChunk(content='Hi', finish_reason=None, usage=None, tool_calls=None)
        >>> _map_stream_chunk({"candidates": [{"content": {"parts": []}}]}) is None
        True
    """
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
    """
    Map a Gemini ``embedContent`` response body into an :class:`EmbeddingsResponse`.

    Args:
        data: The parsed JSON response body from Gemini.
        model: The originally-requested model identifier, passed in
            explicitly since Gemini's ``embedContent`` response doesn't echo
            it back.

    Returns:
        The normalized :class:`~src.providers.base.EmbeddingsResponse`,
        wrapping the single embedding vector Gemini returns (this endpoint
        only ever embeds one piece of content per call) in a one-element
        list to match the shared multi-vector response shape.

    Example:
        >>> _map_embeddings_response(
        ...     {"embedding": {"values": [0.1, 0.2]}},
        ...     model="text-embedding-004",
        ... )
        EmbeddingsResponse(model='text-embedding-004', embeddings=[[0.1, 0.2]], usage=None)
    """
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
    """
    Map a Gemini ``/v1beta/models`` response body into a tuple of :class:`ModelInfo`.

    Every entry's ``name`` field from Gemini is already the full
    ``"models/<id>"`` resource path, which becomes ``ModelInfo.id``; the
    prefix-stripped short form becomes both ``ModelInfo.name`` and
    ``display_name`` — mirroring how :meth:`GeminiProvider.validate_model`
    later needs to match against either form. As with the other adapters,
    Gemini's models endpoint reports no capability flags, so
    ``supports_streaming``/``supports_vision``/``supports_tools``/
    ``supports_json`` are all assumed ``True``, and
    ``supports_embeddings`` is inferred from ``"embed"`` appearing in the
    short model name.

    Args:
        data: The parsed JSON response body from ``GET /v1beta/models``.

    Returns:
        A tuple of :class:`~src.providers.base.ModelInfo`. Entries with no
        usable ``name`` are skipped; a non-list ``data["models"]`` yields an
        empty tuple.

    Example:
        >>> _map_models({"models": [{"name": "models/gemini-pro"}]})
        (ModelInfo(id='models/gemini-pro', name='gemini-pro', display_name='gemini-pro', context_window=None, supports_streaming=True, supports_embeddings=False, supports_vision=True, supports_tools=True, supports_json=True),)
    """
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
    """
    Map Gemini's ``usageMetadata`` object into a :class:`~src.providers.base.TokenUsage`.

    Gemini names its usage fields ``promptTokenCount``/
    ``candidatesTokenCount``/``totalTokenCount`` — unlike Ollama/Anthropic,
    Gemini *does* report a combined total directly, so (unlike those two
    adapters) no total is computed manually here.

    Args:
        raw_usage: The raw ``usageMetadata`` value from a response body.

    Returns:
        The normalized :class:`~src.providers.base.TokenUsage`, or ``None``
        if ``raw_usage`` isn't a mapping, or none of the three fields could
        be parsed.

    Example:
        >>> _map_usage({"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7})
        TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        >>> _map_usage(None) is None
        True
    """
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
    """
    Best-effort parse of a tool-call arguments value into a plain dict.

    Used when replaying a prior assistant tool-call turn back to Gemini
    (see :func:`_map_message`), since
    :class:`~src.providers.base.ToolCall.arguments` is always a string on
    the normalized DTO but Gemini's ``functionCall.args`` expects the parsed
    object directly.

    Args:
        value: The raw arguments value — expected to be either already a
            ``dict`` or a JSON-encoded string.

    Returns:
        The parsed dict if ``value`` was a dict or valid JSON decoding to an
        object; an empty dict otherwise.

    Example:
        >>> _parse_json_object('{"city": "Paris"}')
        {'city': 'Paris'}
        >>> _parse_json_object("")
        {}
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        import json

        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _serialize_json(value: object) -> str:
    """
    Serialize a tool call's parsed arguments back into the normalized string form.

    The inverse of :func:`_parse_json_object`: Gemini's ``functionCall.args``
    arrives as a parsed JSON object, but
    :class:`~src.providers.base.ToolCall.arguments` must be a string.

    Args:
        value: The value to serialize — typically the parsed ``args`` object
            from a Gemini ``functionCall`` part, but passed through
            unchanged if already a string.

    Returns:
        ``value`` unchanged if it's already a string; otherwise its JSON
        serialization.

    Example:
        >>> _serialize_json({"city": "Paris"})
        '{"city": "Paris"}'
    """
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value)


def _optional_str(value: object) -> str | None:
    """
    Coerce a possibly-``None`` value to ``str``, preserving ``None``.

    Args:
        value: Any value, typically from a parsed JSON response.

    Returns:
        ``None`` if ``value`` is ``None``; otherwise ``str(value)``.

    Example:
        >>> _optional_str(None) is None
        True
        >>> _optional_str("STOP")
        'STOP'
    """
    if value is None:
        return None
    return str(value)


def _as_int(value: object) -> int | None:
    """
    Best-effort coercion of a JSON value to ``int``, tolerating bad input.

    Args:
        value: Any value, typically a usage-count field from a parsed JSON
            response.

    Returns:
        The integer value, or ``None`` if ``value`` is ``None`` or cannot be
        converted to ``int``.

    Example:
        >>> _as_int("5")
        5
        >>> _as_int(None) is None
        True
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Registering at import time is what lets `import src.providers` alone make
# this adapter discoverable via the registry — see `src/providers/__init__.py`.
register_provider(GeminiProvider)
