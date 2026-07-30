"""
OpenAI API adapter implementing :class:`~src.providers.base.BaseProvider`.

Translates the gateway's normalized request/response DTOs
(``ChatRequest``/``ChatResponse``/etc.) to and from OpenAI's Chat Completions
and Embeddings API payloads. Also serves as the closest thing to a
"reference implementation" among the cloud adapters, since OpenAI's request/
response shapes (``choices[0].message``, ``tool_calls`` with a nested
``function`` object, SSE streaming with ``data: [DONE]``) are the de facto
convention several other vendors' APIs loosely follow or are commonly
compared against.

Unlike ``OllamaProvider``, this adapter uses
:func:`~src.providers.retry.retry_async` for automatic exponential-backoff
retries on transient failures, and :class:`~src.providers.validation.ModelListCache`
to avoid refetching OpenAI's full model catalog on every request — both
appropriate for a shared, rate-limited, occasionally-flaky cloud API in a way
they aren't for a local Ollama instance.
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
    """
    HTTP adapter for the OpenAI Chat Completions and Embeddings APIs.

    Example:
        >>> import asyncio
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> provider = OpenAIProvider(api_key="sk-test")
        >>> request = ChatRequest(
        ...     model="gpt-4o",
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
        organization: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Create an OpenAI adapter authenticated with the given API key.

        Args:
            api_key: The caller's OpenAI API key. Validated as non-empty via
                :func:`~src.providers.http_mixin.require_non_empty_string`
                so a missing/blank key fails fast at construction time
                rather than surfacing as a confusing 401 on the first
                request.
            base_url: The OpenAI API base URL. Defaults to
                ``"https://api.openai.com/v1"``; overridable to point at an
                OpenAI-compatible proxy or gateway.
            organization: Optional OpenAI organization id to scope requests
                to (see :func:`~src.providers.http_auth.openai_headers`).
                Whitespace is stripped; an empty/whitespace-only value is
                treated the same as ``None``.
            timeout: Request timeout in seconds for the internally created
                HTTP client, applied only when ``http_client`` is ``None``.
            http_client: An optional pre-built ``httpx.AsyncClient`` to reuse.
            max_retries: Maximum retry attempts for
                :func:`~src.providers.retry.retry_async` on transient
                failures. Defaults to
                ``Settings.PROVIDER_HTTP_MAX_RETRIES`` when not explicitly
                provided, so the retry budget is centrally configurable
                without needing to change adapter code, while still letting
                callers (e.g. tests) override it per instance.

        Raises:
            ValueError: If ``api_key`` is empty or only whitespace.
        """
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
        """
        Execute a non-streaming chat completion via ``POST /chat/completions``.

        The HTTP call is wrapped in :func:`~src.providers.retry.retry_async`
        so transient failures (rate limiting, brief 5xx blips) are retried
        with exponential backoff before this method gives up and raises.

        Args:
            request: The normalized chat request to execute.

        Returns:
            The normalized chat completion response, mapped via
            :func:`_map_chat_response`.

        Raises:
            ModelNotFoundError: HTTP 404 (invalid/unavailable model).
            InvalidRequestError: HTTP 400 (malformed request/parameters).
            AuthenticationError: HTTP 401/403 (invalid/missing API key).
            RateLimitError: HTTP 429, after retries are exhausted.
            ProviderUnavailableError: HTTP 5xx after retries are exhausted,
                or a transport-level failure (connection error, timeout)
                after retries are exhausted.
            ProviderError: Any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import ChatMessage, ChatRequest
            >>> provider = OpenAIProvider(api_key="sk-test")
            >>> request = ChatRequest(
            ...     model="gpt-4o",
            ...     messages=[ChatMessage(role="user", content="Hi")],
            ... )
            >>> asyncio.run(provider.chat(request))  # doctest: +SKIP
            ChatResponse(content='Hello!', model='gpt-4o', ...)
        """
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
        """
        Stream chat completion chunks via SSE from ``POST /chat/completions``.

        Note the streaming path does *not* go through
        :func:`~src.providers.retry.retry_async` the way :meth:`chat` does:
        once the SSE stream has started, partial content may already have
        been yielded to the caller, so transparently retrying the whole
        request would risk duplicating output. Retries here are effectively
        limited to whatever happens before the stream is opened (i.e. the
        initial connection/response headers).

        Args:
            request: The normalized chat request to execute, typically with
                ``request.stream`` set to ``True``.

        Yields:
            :class:`~src.providers.base.ChatStreamChunk` objects, one per SSE
            ``data:`` event, parsed via
            :func:`~src.providers.streaming.iter_sse_json` and mapped by
            :func:`_map_stream_chunk`.

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
            ...     provider = OpenAIProvider(api_key="sk-test")
            ...     request = ChatRequest(
            ...         model="gpt-4o",
            ...         messages=[ChatMessage(role="user", content="Hi")],
            ...         stream=True,
            ...     )
            ...     async for chunk in provider.stream_chat(request):
            ...         print(chunk.content, end="")
            >>> asyncio.run(demo())  # doctest: +SKIP
        """
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
        """
        Generate embeddings via ``POST /embeddings``.

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
            >>> provider = OpenAIProvider(api_key="sk-test")
            >>> request = EmbeddingsRequest(model="text-embedding-3-small", input="hello")
            >>> asyncio.run(provider.embeddings(request))  # doctest: +SKIP
            EmbeddingsResponse(model='text-embedding-3-small', embeddings=[[...]], usage=...)
        """
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
        """
        List available models via ``GET /models``, using a short-lived cache.

        Checks :attr:`_model_cache` first and only calls OpenAI's API on a
        cache miss (see :class:`~src.providers.validation.ModelListCache`),
        since OpenAI's model catalog changes infrequently relative to how
        often it might otherwise be queried (e.g. once per incoming chat
        request, to validate the requested model).

        Returns:
            A tuple of :class:`~src.providers.base.ModelInfo`, either served
            from cache or freshly fetched and mapped via :func:`_map_models`.

        Raises:
            AuthenticationError: HTTP 401/403.
            RateLimitError: HTTP 429, after retries are exhausted.
            ProviderUnavailableError: HTTP 5xx after retries are exhausted,
                or a transport-level failure after retries are exhausted.
            ProviderError: Any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> provider = OpenAIProvider(api_key="sk-test")
            >>> asyncio.run(provider.list_models())  # doctest: +SKIP
            (ModelInfo(id='gpt-4o', name='gpt-4o', ...), ...)
        """
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
        """
        Return whether ``model`` is offered by OpenAI, using the cached catalog.

        Overrides :meth:`~src.providers.base.BaseProvider.validate_model` to
        check :attr:`_model_cache` directly (falling back to a full
        :meth:`list_models` call only on a cache miss) rather than always
        calling :meth:`list_models`, which itself already checks the cache —
        functionally equivalent, but this avoids the extra method-call
        indirection and clarifies the intent at this call site.

        Args:
            model: The model identifier to check, e.g. ``"gpt-4o"``.

        Returns:
            ``True`` if ``model`` matches any cached/fetched model's ``id``
            or ``name``; ``False`` otherwise.

        Raises:
            ProviderError: If a cache miss forces a :meth:`list_models` call
                that itself fails (see :meth:`list_models` for the specific
                subclasses raised).

        Example:
            >>> import asyncio
            >>> provider = OpenAIProvider(api_key="sk-test")
            >>> asyncio.run(provider.validate_model("gpt-4o"))  # doctest: +SKIP
            True
        """
        cached = self._model_cache.get(_PROVIDER)
        if cached is None:
            cached = await self.list_models()
        return any(item.id == model or item.name == model for item in cached)

    async def health_check(self) -> HealthCheckResult:
        """
        Probe OpenAI readiness via ``GET /models``.

        Connection and HTTP failures are encoded as unhealthy results rather
        than raised exceptions, matching
        :meth:`~src.providers.base.BaseProvider.health_check`'s contract.
        Deliberately does not go through :func:`~src.providers.retry.retry_async`
        or the model cache — a health check should reflect the provider's
        *current* reachability on this single attempt, not a cached or
        retried view of it.

        Returns:
            A :class:`~src.providers.base.HealthCheckResult` describing
            whether OpenAI responded successfully, with the reported model
            count when healthy.

        Raises:
            (none) — every failure path is caught and encoded into the
            returned result.

        Example:
            >>> import asyncio
            >>> provider = OpenAIProvider(api_key="sk-test")
            >>> asyncio.run(provider.health_check())  # doctest: +SKIP
            HealthCheckResult(healthy=True, ...)
        """
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
    """
    Build the JSON body for OpenAI's ``POST /chat/completions`` endpoint.

    Only includes optional fields (``temperature``, ``max_tokens``,
    ``top_p``, ``stop``, ``tools``, ``tool_choice``, ``response_format``)
    when they are actually set on ``request``, so the payload sent to OpenAI
    matches exactly what the caller asked for and lets OpenAI apply its own
    defaults for anything unspecified rather than the gateway silently
    forcing a value.

    Args:
        request: The normalized chat request to translate.
        stream: Whether this payload is for :meth:`OpenAIProvider.stream_chat`
            (``True``) or :meth:`OpenAIProvider.chat` (``False``).

    Returns:
        A JSON-serializable dict matching OpenAI's Chat Completions request
        schema.

    Example:
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> request = ChatRequest(
        ...     model="gpt-4o",
        ...     messages=[ChatMessage(role="user", content="Hi")],
        ...     temperature=0.7,
        ... )
        >>> _build_chat_payload(request, stream=False)
        {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': False, 'temperature': 0.7}
    """
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
    """
    Map a normalized :class:`~src.providers.base.ChatMessage` to OpenAI's message shape.

    Handles the three distinct message "modes" a :class:`ChatMessage` can be
    in (checked in priority order, since they are mutually exclusive in
    practice): a tool-result turn (``tool_call_id`` set), an
    assistant-tool-call turn (``tool_calls`` set), or a regular content turn
    (plain text or multimodal ``parts``).

    Args:
        message: The message to translate.

    Returns:
        A dict matching one of OpenAI's three message shapes: a
        ``{"role": "tool", "tool_call_id", "content"}`` result message, an
        assistant message with a ``tool_calls`` list (using ``None`` for
        ``content`` when empty, since OpenAI expects ``null`` rather than an
        empty string for tool-call-only assistant turns), or a plain
        ``{"role", "content"}`` message.

    Example:
        >>> _map_message(ChatMessage(role="user", content="Hi"))
        {'role': 'user', 'content': 'Hi'}
        >>> _map_message(ChatMessage(role="tool", tool_call_id="call_1", content="42"))
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '42'}
    """
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
    """
    Map a message's content to either a plain string or an OpenAI content-block list.

    OpenAI's API accepts ``content`` as either a plain string (for
    text-only messages) or a list of typed content blocks (for multimodal
    messages). This function returns the plain-string form whenever
    ``message.parts`` is empty, and only switches to the list form when
    multimodal parts are present — since sending every message as a
    single-element content-block list would work but is unnecessarily
    verbose compared to matching OpenAI's simpler text-only convention.

    Args:
        message: The message whose content is being mapped.

    Returns:
        ``message.content`` unchanged if there are no ``parts``; otherwise a
        list of content blocks built from ``message.parts`` (via
        :func:`_map_content_part`), with any leading plain-text
        ``message.content`` prepended as its own ``{"type": "text", ...}``
        block.

    Example:
        >>> _map_message_content(ChatMessage(role="user", content="Hi"))
        'Hi'
        >>> _map_message_content(
        ...     ChatMessage(role="user", content="Look:", parts=[ImagePart(url="https://x/y.jpg")])
        ... )
        [{'type': 'text', 'text': 'Look:'}, {'type': 'image_url', 'image_url': {'url': 'https://x/y.jpg'}}]
    """
    if message.parts:
        blocks: list[dict[str, Any]] = []
        for part in message.parts:
            blocks.extend(_map_content_part(part))
        if message.content:
            blocks.insert(0, {"type": "text", "text": message.content})
        return blocks
    return message.content


def _map_content_part(part: ContentPart) -> list[dict[str, Any]]:
    """
    Map one :data:`~src.providers.base.ContentPart` to OpenAI content-block(s).

    Args:
        part: A single text or image content part.

    Returns:
        A list containing the mapped content block (a single-element list
        for a valid part), or an empty list if ``part`` is an
        :class:`~src.providers.base.ImagePart` with neither ``url`` nor
        ``base64_data`` set (nothing usable to send). Returning a list
        (rather than a single dict or ``None``) lets the caller
        unconditionally ``blocks.extend(...)`` regardless of whether zero or
        one block resulted.

    Example:
        >>> _map_content_part(TextPart(text="hi"))
        [{'type': 'text', 'text': 'hi'}]
        >>> _map_content_part(ImagePart(base64_data="aGk=", media_type="image/png"))
        [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGk='}}]
    """
    if isinstance(part, TextPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, ImagePart):
        if part.url:
            return [{"type": "image_url", "image_url": {"url": part.url}}]
        if part.base64_data:
            # OpenAI accepts inline images as a `data:` URL rather than a
            # separate base64 field, so the media type + base64 payload are
            # combined into one URL string here.
            data_url = f"data:{part.media_type};base64,{part.base64_data}"
            return [{"type": "image_url", "image_url": {"url": data_url}}]
    return []


def _map_tool(tool: ToolDefinition) -> dict[str, Any]:
    """
    Map a :class:`~src.providers.base.ToolDefinition` to OpenAI's function-tool schema.

    Args:
        tool: The tool definition to translate.

    Returns:
        A dict of the form ``{"type": "function", "function": {"name", ...}}``,
        OpenAI's wrapper shape for function tools. ``description`` and
        ``parameters`` are only included when present on ``tool``, letting
        OpenAI apply its own defaults (e.g. "no parameters") otherwise.

    Example:
        >>> _map_tool(ToolDefinition(name="get_weather", description="Get weather."))
        {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get weather.'}}
    """
    function: dict[str, Any] = {"name": tool.name}
    if tool.description:
        function["description"] = tool.description
    if tool.parameters is not None:
        function["parameters"] = dict(tool.parameters)
    return {"type": "function", "function": function}


def _map_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    """
    Map a :class:`~src.providers.base.ResponseFormat` to OpenAI's ``response_format`` field.

    Args:
        response_format: The structured-output configuration to translate.

    Returns:
        ``{"type": "json_schema", "json_schema": {...}}`` when a schema is
        specified; ``{"type": "json_object"}`` for unconstrained JSON mode
        (OpenAI's historical name for plain-JSON mode, distinct from its
        newer schema-constrained mode); or ``{"type": "text"}`` for
        unconstrained plain text, which is also the fallback returned for a
        ``"json_schema"`` request that is missing its schema (treated as if
        no structured output was requested at all, rather than sending an
        incomplete/invalid ``response_format`` to OpenAI).

    Example:
        >>> _map_response_format(ResponseFormat(type="json"))
        {'type': 'json_object'}
        >>> _map_response_format(ResponseFormat(type="text"))
        {'type': 'text'}
    """
    if response_format.type == "json_schema" and response_format.json_schema is not None:
        return {
            "type": "json_schema",
            "json_schema": dict(response_format.json_schema),
        }
    if response_format.type == "json":
        return {"type": "json_object"}
    return {"type": "text"}


def _map_chat_response(data: Mapping[str, Any]) -> ChatResponse:
    """
    Map an OpenAI Chat Completions response body into a :class:`ChatResponse`.

    Reads only ``choices[0]`` — OpenAI can technically return multiple
    completion choices per request (via the ``n`` parameter, not currently
    exposed on :class:`~src.providers.base.ChatRequest`), but since this
    gateway never requests more than one, only the first is mapped.

    Args:
        data: The parsed JSON response body from OpenAI.

    Returns:
        The normalized :class:`~src.providers.base.ChatResponse`, with
        ``choices`` defaulting to an empty dict/list at each level so a
        malformed or unexpectedly-shaped response degrades to empty/``None``
        fields rather than raising a ``KeyError``/``IndexError``.

    Example:
        >>> _map_chat_response({
        ...     "id": "chatcmpl-1",
        ...     "model": "gpt-4o",
        ...     "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
        ...     "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        ... })
        ChatResponse(content='Hi!', model='gpt-4o', finish_reason='stop', usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7), provider_response_id='chatcmpl-1', tool_calls=None, details={})
    """
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
    """
    Map one OpenAI SSE ``data:`` payload into a :class:`ChatStreamChunk`.

    Streaming responses use ``choices[0].delta`` (an incremental patch)
    rather than the full ``choices[0].message`` object used in non-streaming
    responses — the same logical fields (content, tool calls) exist, just
    under a different key and representing only the *new* content since the
    last chunk rather than the full accumulated content.

    Args:
        data: One parsed JSON SSE payload.

    Returns:
        The normalized :class:`~src.providers.base.ChatStreamChunk` for this
        chunk. ``usage`` is only populated for chunks that include it (some
        vendors — and OpenAI, depending on ``stream_options`` — only send
        usage data on a final chunk once streaming completes).

    Example:
        >>> _map_stream_chunk({"choices": [{"delta": {"content": "Hi"}}]})
        ChatStreamChunk(content='Hi', finish_reason=None, usage=None, tool_calls=None)
    """
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
    """
    Map OpenAI's ``tool_calls`` array into a tuple of :class:`~src.providers.base.ToolCall`.

    Args:
        raw_calls: The raw ``tool_calls`` value from a message or delta,
            expected to be a list of ``{"id", "function": {"name",
            "arguments"}}`` objects but not guaranteed to be well-formed.

    Returns:
        A tuple of :class:`~src.providers.base.ToolCall` built from
        well-formed entries only (both a non-empty ``id`` and ``name`` are
        required — a malformed entry missing either is silently dropped
        rather than producing an unusable partial ``ToolCall``); ``None`` if
        ``raw_calls`` isn't a non-empty list, or if every entry was dropped
        as malformed. Returning ``None`` (rather than an empty tuple) here
        matches the "no tool calls" representation used throughout the
        normalized DTOs.

    Example:
        >>> _map_tool_calls([
        ...     {"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}},
        ... ])
        (ToolCall(id='call_1', name='get_weather', arguments='{}'),)
        >>> _map_tool_calls(None) is None
        True
    """
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
    """
    Map an OpenAI ``/embeddings`` response body into an :class:`EmbeddingsResponse`.

    Args:
        data: The parsed JSON response body from OpenAI.

    Returns:
        The normalized :class:`~src.providers.base.EmbeddingsResponse`, with
        malformed entries in ``data["data"]`` silently skipped rather than
        raising.

    Example:
        >>> _map_embeddings_response({
        ...     "model": "text-embedding-3-small",
        ...     "data": [{"embedding": [0.1, 0.2]}],
        ...     "usage": {"prompt_tokens": 3, "total_tokens": 3},
        ... })
        EmbeddingsResponse(model='text-embedding-3-small', embeddings=[[0.1, 0.2]], usage=TokenUsage(prompt_tokens=3, completion_tokens=None, total_tokens=3))
    """
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
    """
    Map an OpenAI ``/models`` response body into a tuple of :class:`ModelInfo`.

    OpenAI's ``/models`` endpoint returns only bare model ids with no
    capability metadata at all, so every ``supports_*`` flag here is
    inferred from naming conventions in the model id string:

    - ``supports_embeddings``: ids starting with ``"text-embedding"``.
    - ``supports_vision``: ids containing ``"vision"`` or starting with
      ``"gpt-4"`` (GPT-4-family models are generally multimodal; this is a
      simplification that may over-include some GPT-4 variants without
      vision support).
    - ``supports_tools`` / ``supports_json``: any id starting with
      ``"gpt-"``, since function calling and JSON mode are broadly available
      across the GPT model family (as opposed to e.g. legacy completion-only
      or embedding-only models).
    - ``supports_streaming=True`` unconditionally, since OpenAI's Chat
      Completions endpoint supports streaming for every chat-capable model.

    Args:
        data: The parsed JSON response body from ``GET /models``.

    Returns:
        A tuple of :class:`~src.providers.base.ModelInfo`. Entries with no
        usable ``id`` are skipped; a non-list ``data["data"]`` yields an
        empty tuple.

    Example:
        >>> _map_models({"data": [{"id": "gpt-4o"}]})
        (ModelInfo(id='gpt-4o', name='gpt-4o', display_name='gpt-4o', context_window=None, supports_streaming=True, supports_embeddings=False, supports_vision=True, supports_tools=True, supports_json=True),)
    """
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
    """
    Map OpenAI's ``usage`` object into a :class:`~src.providers.base.TokenUsage`.

    Args:
        raw_usage: The raw ``usage`` value from a response body, expected to
            be a mapping with ``prompt_tokens``/``completion_tokens``/
            ``total_tokens`` keys but not guaranteed to be well-formed.

    Returns:
        The normalized :class:`~src.providers.base.TokenUsage`, or ``None``
        if ``raw_usage`` isn't a mapping at all, or none of the three usage
        fields could be parsed as integers.

    Example:
        >>> _map_usage({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        >>> _map_usage(None) is None
        True
    """
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
    """
    Coerce a possibly-``None`` value to ``str``, preserving ``None``.

    See :func:`src.providers.ollama._optional_str` for the same helper
    pattern; duplicated locally (rather than imported) since these small
    response-mapping helpers are treated as private, module-local utilities
    rather than shared infrastructure.

    Args:
        value: Any value, typically from a parsed JSON response.

    Returns:
        ``None`` if ``value`` is ``None``; otherwise ``str(value)``.

    Example:
        >>> _optional_str(None) is None
        True
        >>> _optional_str("stop")
        'stop'
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
        >>> _as_int("7")
        7
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
register_provider(OpenAIProvider)
