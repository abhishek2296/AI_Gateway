"""
Anthropic Messages API adapter implementing :class:`~src.providers.base.BaseProvider`.

Translates the gateway's normalized request/response DTOs to and from
Anthropic's Messages API payloads. Anthropic's API diverges from the
OpenAI-style convention in several notable ways that this module has to
account for:

- System prompts are a dedicated top-level ``system`` string field, not a
  ``{"role": "system", ...}`` message in the ``messages`` array — so system
  messages must be extracted out of the normalized message list before
  building the payload (see :func:`_split_system_messages`).
- ``max_tokens`` is a *required* field on every request (unlike OpenAI/
  Gemini, where it's optional), so a fallback value is substituted when the
  caller didn't specify one.
- Tool calls/results are represented as typed content blocks
  (``tool_use``/``tool_result``) embedded within a message's ``content``
  array, rather than as separate top-level message fields.
- Streaming uses SSE with paired ``event:``/``data:`` lines (see
  :func:`~src.providers.streaming.iter_sse_events`), where the event type
  determines how to interpret the payload.
- Anthropic has no embeddings endpoint at all — see :meth:`AnthropicProvider.embeddings`.
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
    """
    HTTP adapter for the Anthropic Messages API.

    Example:
        >>> import asyncio
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> provider = AnthropicProvider(api_key="sk-ant-test")
        >>> request = ChatRequest(
        ...     model="claude-3-5-sonnet-20241022",
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
        api_version: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Create an Anthropic adapter authenticated with the given API key.

        Args:
            api_key: The caller's Anthropic API key. Validated as non-empty
                via :func:`~src.providers.http_mixin.require_non_empty_string`.
            base_url: The Anthropic API base URL. Defaults to
                ``"https://api.anthropic.com"``.
            api_version: The Anthropic API version string to pin requests to
                (see :func:`~src.providers.http_auth.anthropic_headers`).
                Defaults to ``Settings.ANTHROPIC_API_VERSION`` when not
                explicitly provided, so the pinned version is centrally
                configurable rather than hardcoded per call site.
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
        """
        Execute a non-streaming chat completion via ``POST /v1/messages``.

        The HTTP call is wrapped in :func:`~src.providers.retry.retry_async`
        for automatic exponential-backoff retries on transient failures.

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
            >>> provider = AnthropicProvider(api_key="sk-ant-test")
            >>> request = ChatRequest(
            ...     model="claude-3-5-sonnet-20241022",
            ...     messages=[ChatMessage(role="user", content="Hi")],
            ... )
            >>> asyncio.run(provider.chat(request))  # doctest: +SKIP
            ChatResponse(content='Hello!', model='claude-3-5-sonnet-20241022', ...)
        """
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
        """
        Stream chat completion chunks via SSE from ``POST /v1/messages``.

        Anthropic's stream is a sequence of distinct, named events (e.g.
        ``message_start``, ``content_block_start``,
        ``content_block_delta``, ``message_delta``, ``message_stop``), of
        which only ``content_block_delta`` (incremental text) and
        ``message_delta`` (finish reason + usage) currently produce a
        yielded chunk — see :func:`_map_stream_event`, which returns
        ``None`` for every other event type so this method can filter them
        out.

        Args:
            request: The normalized chat request to execute.

        Yields:
            :class:`~src.providers.base.ChatStreamChunk` objects for each
            ``content_block_delta``/``message_delta`` event, parsed via
            :func:`~src.providers.streaming.iter_sse_events` and mapped by
            :func:`_map_stream_event`.

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
            ...     provider = AnthropicProvider(api_key="sk-ant-test")
            ...     request = ChatRequest(
            ...         model="claude-3-5-sonnet-20241022",
            ...         messages=[ChatMessage(role="user", content="Hi")],
            ...         stream=True,
            ...     )
            ...     async for chunk in provider.stream_chat(request):
            ...         print(chunk.content, end="")
            >>> asyncio.run(demo())  # doctest: +SKIP
        """
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
        """
        Always raise: Anthropic's Messages API has no embeddings endpoint.

        Args:
            request: The embeddings request (unused — this method never
                attempts a call, since there is no endpoint to call).

        Returns:
            This method never returns; it always raises.

        Raises:
            UnsupportedCapabilityError: Unconditionally, on every call.

        Example:
            >>> import asyncio
            >>> provider = AnthropicProvider(api_key="sk-ant-test")
            >>> request = EmbeddingsRequest(model="n/a", input="hello")
            >>> asyncio.run(provider.embeddings(request))
            Traceback (most recent call last):
                ...
            UnsupportedCapabilityError: Anthropic does not expose an embeddings API.
        """
        raise UnsupportedCapabilityError(
            "Anthropic does not expose an embeddings API.",
            provider=_PROVIDER,
        )

    async def list_models(self) -> Sequence[ModelInfo]:
        """
        List available models via ``GET /v1/models``, using a short-lived cache.

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
            >>> provider = AnthropicProvider(api_key="sk-ant-test")
            >>> asyncio.run(provider.list_models())  # doctest: +SKIP
            (ModelInfo(id='claude-3-5-sonnet-20241022', ...), ...)
        """
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
        """
        Return whether ``model`` is offered by Anthropic, using the cached catalog.

        Overrides :meth:`~src.providers.base.BaseProvider.validate_model` to
        check :attr:`_model_cache` directly before falling back to
        :meth:`list_models` on a cache miss (same rationale as
        ``OpenAIProvider.validate_model``).

        Args:
            model: The model identifier to check, e.g.
                ``"claude-3-5-sonnet-20241022"``.

        Returns:
            ``True`` if ``model`` matches any cached/fetched model's ``id``
            or ``name``; ``False`` otherwise.

        Raises:
            ProviderError: If a cache miss forces a :meth:`list_models` call
                that itself fails.

        Example:
            >>> import asyncio
            >>> provider = AnthropicProvider(api_key="sk-ant-test")
            >>> asyncio.run(provider.validate_model("claude-3-5-sonnet-20241022"))  # doctest: +SKIP
            True
        """
        cached = self._model_cache.get(_PROVIDER)
        if cached is None:
            cached = await self.list_models()
        return any(item.id == model or item.name == model for item in cached)

    async def health_check(self) -> HealthCheckResult:
        """
        Probe Anthropic readiness via ``GET /v1/models``.

        Connection and HTTP failures are encoded as unhealthy results rather
        than raised exceptions.

        Returns:
            A :class:`~src.providers.base.HealthCheckResult` describing
            whether Anthropic responded successfully, with the reported
            model count when healthy.

        Raises:
            (none) — every failure path is caught and encoded into the
            returned result.

        Example:
            >>> import asyncio
            >>> provider = AnthropicProvider(api_key="sk-ant-test")
            >>> asyncio.run(provider.health_check())  # doctest: +SKIP
            HealthCheckResult(healthy=True, ...)
        """
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
    """
    Build the JSON body for Anthropic's ``POST /v1/messages`` endpoint.

    Unlike OpenAI/Gemini, Anthropic requires ``max_tokens`` on every
    request — there is no server-side default the API will apply if it's
    omitted, so this function substitutes ``1024`` as a reasonable fallback
    when the caller's :class:`~src.providers.base.ChatRequest` didn't
    specify one, rather than sending an invalid request that Anthropic would
    reject outright.

    Args:
        request: The normalized chat request to translate.
        stream: Whether this payload is for
            :meth:`AnthropicProvider.stream_chat` (``True``) or
            :meth:`AnthropicProvider.chat` (``False``, the default).

    Returns:
        A JSON-serializable dict matching Anthropic's Messages API request
        schema, with any system-role messages extracted into the top-level
        ``system`` field (see :func:`_split_system_messages`) rather than
        left in ``messages``.

    Example:
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> request = ChatRequest(
        ...     model="claude-3-5-sonnet-20241022",
        ...     messages=[
        ...         ChatMessage(role="system", content="Be terse."),
        ...         ChatMessage(role="user", content="Hi"),
        ...     ],
        ... )
        >>> _build_messages_payload(request)
        {'model': 'claude-3-5-sonnet-20241022', 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 1024, 'stream': False, 'system': 'Be terse.'}
    """
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
    """
    Separate ``system``-role messages out of a message list for Anthropic's payload shape.

    Anthropic expects system instructions as a single top-level ``system``
    string, not as messages interleaved in the ``messages`` array the way
    OpenAI/Gemini-style APIs allow. If the normalized request contains
    multiple system messages (uncommon, but not disallowed by the
    normalized :class:`~src.providers.base.ChatRequest` contract), they are
    joined with a blank-line separator into one combined string, preserving
    their relative order.

    Args:
        messages: The full, normalized message list from a
            :class:`~src.providers.base.ChatRequest`.

    Returns:
        A tuple of ``(system_prompt, remaining_messages)``, where
        ``system_prompt`` is the joined system content (or ``None`` if there
        were no system messages, or all of them had empty content), and
        ``remaining_messages`` is every non-system message in its original
        order.

    Example:
        >>> _split_system_messages([
        ...     ChatMessage(role="system", content="Be terse."),
        ...     ChatMessage(role="user", content="Hi"),
        ... ])
        ('Be terse.', [ChatMessage(role='user', content='Hi', parts=None, tool_calls=None, tool_call_id=None)])
        >>> _split_system_messages([ChatMessage(role="user", content="Hi")])
        (None, [ChatMessage(role='user', content='Hi', parts=None, tool_calls=None, tool_call_id=None)])
    """
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
    """
    Map a normalized :class:`~src.providers.base.ChatMessage` to Anthropic's message shape.

    Anthropic represents both tool results and tool calls as typed content
    blocks embedded in a message's ``content`` array (``tool_result``/
    ``tool_use``), rather than as separate top-level message fields the way
    OpenAI does. Notably, a tool-result message is sent with
    ``role="user"`` (not ``role="tool"``, which Anthropic's API doesn't
    recognize) — Anthropic models tool results as part of the *user's* turn
    in the conversation.

    Args:
        message: The message to translate.

    Returns:
        A dict shaped as one of: a ``role="user"`` message wrapping a
        ``tool_result`` content block, a message with mixed text +
        ``tool_use`` blocks (for an assistant's tool-call turn), or a
        regular content message (plain text or multimodal blocks).

    Example:
        >>> _map_message(ChatMessage(role="user", content="Hi"))
        {'role': 'user', 'content': 'Hi'}
        >>> _map_message(ChatMessage(role="tool", tool_call_id="call_1", content="42"))
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1', 'content': '42'}]}
    """
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
    """
    Map a message's content to either a plain string or an Anthropic content-block list.

    Mirrors the same "plain string when possible, block list only when
    multimodal parts exist" strategy used by the OpenAI adapter's
    ``_map_message_content`` — Anthropic likewise accepts either
    representation for text-only content.

    Args:
        message: The message whose content is being mapped.

    Returns:
        ``message.content`` unchanged if there are no ``parts``; otherwise a
        list of content blocks built from ``message.parts`` (via
        :func:`_map_content_part`), with any leading plain-text
        ``message.content`` prepended as a ``{"type": "text", ...}`` block.

    Example:
        >>> _map_message_content(ChatMessage(role="user", content="Hi"))
        'Hi'
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
    Map one :data:`~src.providers.base.ContentPart` to Anthropic content-block(s).

    Anthropic's image blocks only support inline base64 data (no ``url``
    field, unlike OpenAI's ``image_url``), so an :class:`ImagePart` supplied
    via URL cannot be sent as a native image block. Rather than silently
    dropping the image or raising, it's degraded to a plain text block
    describing the URL (``"[image:<url>]"``) — a lossy but harmless
    fallback that at least preserves *some* signal to the model about a
    referenced image, instead of losing it entirely.

    Args:
        part: A single text or image content part.

    Returns:
        A list containing the mapped content block, or an empty list if
        ``part`` is an :class:`ImagePart` with neither ``base64_data`` nor
        ``url`` set.

    Example:
        >>> _map_content_part(TextPart(text="hi"))
        [{'type': 'text', 'text': 'hi'}]
        >>> _map_content_part(ImagePart(base64_data="aGk=", media_type="image/png"))
        [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'aGk='}}]
        >>> _map_content_part(ImagePart(url="https://example.com/x.jpg"))
        [{'type': 'text', 'text': '[image:https://example.com/x.jpg]'}]
    """
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
    """
    Map a :class:`~src.providers.base.ToolDefinition` to Anthropic's tool schema.

    Anthropic's flatter tool schema (``{"name", "input_schema", ...}``, no
    nested ``"function"`` wrapper the way OpenAI/Gemini use) requires
    ``input_schema`` to always be present, so an empty
    ``{"type": "object", "properties": {}}`` schema is substituted when the
    tool definition doesn't specify parameters, since Anthropic's API
    doesn't accept an entirely missing schema.

    Args:
        tool: The tool definition to translate.

    Returns:
        A dict of the form ``{"name", "input_schema", ["description"]}``.

    Example:
        >>> _map_tool(ToolDefinition(name="get_weather"))
        {'name': 'get_weather', 'input_schema': {'type': 'object', 'properties': {}}}
    """
    definition: dict[str, Any] = {
        "name": tool.name,
        "input_schema": dict(tool.parameters or {"type": "object", "properties": {}}),
    }
    if tool.description:
        definition["description"] = tool.description
    return definition


def _map_chat_response(data: Mapping[str, Any]) -> ChatResponse:
    """
    Map an Anthropic Messages API response body into a :class:`ChatResponse`.

    Anthropic's response ``content`` is a list of typed blocks (``text``,
    ``tool_use``, and — for extended-thinking-enabled models —
    ``thinking``), which this function walks and buckets by type: text
    blocks are concatenated into the final ``content`` string, ``tool_use``
    blocks become :class:`~src.providers.base.ToolCall` entries, and
    ``thinking`` blocks are joined and surfaced via ``ChatResponse.details``
    under the ``"thinking"`` key, since extended reasoning output has no
    dedicated field on the normalized DTO.

    Args:
        data: The parsed JSON response body from Anthropic.

    Returns:
        The normalized :class:`~src.providers.base.ChatResponse`.

    Example:
        >>> _map_chat_response({
        ...     "id": "msg_1",
        ...     "model": "claude-3-5-sonnet-20241022",
        ...     "content": [{"type": "text", "text": "Hi!"}],
        ...     "stop_reason": "end_turn",
        ...     "usage": {"input_tokens": 5, "output_tokens": 2},
        ... })
        ChatResponse(content='Hi!', model='claude-3-5-sonnet-20241022', finish_reason='end_turn', usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7), provider_response_id='msg_1', tool_calls=None, details={})
    """
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
    """
    Map one Anthropic SSE ``(event_type, data)`` pair into a chunk, or ``None`` to skip it.

    Anthropic's stream includes many event types beyond content deltas
    (``message_start``, ``content_block_start``, ``content_block_stop``,
    ``message_stop``, etc.) that carry no incremental content the gateway's
    normalized :class:`~src.providers.base.ChatStreamChunk` needs to
    represent — this function returns ``None`` for all of those, and the
    caller (:meth:`AnthropicProvider.stream_chat`) filters ``None`` results
    out before yielding.

    Only two event types produce a chunk:

    - ``content_block_delta`` with a ``text_delta``: incremental generated
      text.
    - ``content_block_delta`` with a ``thinking_delta``: incremental
      extended-thinking text, surfaced via ``details["thinking_delta"]``
      rather than ``content`` (mirroring how :func:`_map_chat_response`
      keeps "thinking" output separate from the main response text).
    - ``message_delta``: carries the final ``stop_reason`` and cumulative
      usage once generation is complete.

    Args:
        event_type: The SSE ``event:`` value paired with ``data`` (see
            :func:`~src.providers.streaming.iter_sse_events`), or ``None``
            if no event type was present.
        data: The parsed JSON payload paired with ``event_type``.

    Returns:
        A :class:`~src.providers.base.ChatStreamChunk` for a recognized,
        content-bearing event; ``None`` for every other event type (or for a
        recognized event type whose ``delta`` doesn't match a known delta
        type).

    Example:
        >>> _map_stream_event(
        ...     "content_block_delta",
        ...     {"delta": {"type": "text_delta", "text": "Hi"}},
        ... )
        ChatStreamChunk(content='Hi', finish_reason=None, usage=None, tool_calls=None)
        >>> _map_stream_event("content_block_start", {}) is None
        True
    """
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
    """
    Map an Anthropic ``/v1/models`` response body into a tuple of :class:`ModelInfo`.

    Anthropic's models endpoint doesn't report capability flags either, so
    every Claude model returned is assumed to support streaming, vision,
    tools, and JSON output unconditionally — a reasonable simplification
    since (unlike OpenAI's much broader model catalog spanning many
    generations and use cases) essentially every current Claude model
    exposed by this endpoint supports the full modern feature set.

    Args:
        data: The parsed JSON response body from ``GET /v1/models``.

    Returns:
        A tuple of :class:`~src.providers.base.ModelInfo`. Entries with no
        usable ``id`` are skipped; a non-list ``data["data"]`` yields an
        empty tuple.

    Example:
        >>> _map_models({"data": [{"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"}]})
        (ModelInfo(id='claude-3-5-sonnet-20241022', name='claude-3-5-sonnet-20241022', display_name='Claude 3.5 Sonnet', context_window=None, supports_streaming=True, supports_embeddings=False, supports_vision=True, supports_tools=True, supports_json=True),)
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
    """
    Map Anthropic's ``usage`` object into a :class:`~src.providers.base.TokenUsage`.

    Anthropic names its usage fields ``input_tokens``/``output_tokens``
    (rather than OpenAI's ``prompt_tokens``/``completion_tokens``) and does
    not report a combined total, so it's computed here as the sum of the two
    — using ``(prompt or 0) + (completion or 0)`` rather than requiring both
    to be present, since a partial usage report (e.g. only prompt tokens
    known) should still yield a best-effort total rather than ``None``.

    Args:
        raw_usage: The raw ``usage`` value from a response body.

    Returns:
        The normalized :class:`~src.providers.base.TokenUsage`, or ``None``
        if ``raw_usage`` isn't a mapping, or neither ``input_tokens`` nor
        ``output_tokens`` could be parsed.

    Example:
        >>> _map_usage({"input_tokens": 5, "output_tokens": 2})
        TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        >>> _map_usage(None) is None
        True
    """
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
    """
    Best-effort parse of a tool-call arguments value into a plain dict.

    :class:`~src.providers.base.ToolCall.arguments` is always a string on
    the normalized DTO (see its docstring for why), but Anthropic's
    ``tool_use`` block expects the parsed object directly under ``"input"``
    — so when replaying a prior assistant tool-call turn back to Anthropic
    (see :func:`_map_message`), the string arguments must be parsed back
    into a dict.

    The ``import json`` here is local to the function (mirroring
    ``base.estimate_tokens``'s local-import pattern) purely because this is
    the only function in the module that needs it, keeping the module-level
    import list focused on what's used broadly across the file.

    Args:
        value: The raw arguments value — expected to be either already a
            ``dict`` (if constructed programmatically rather than from a
            vendor response) or a JSON-encoded string.

    Returns:
        The parsed dict if ``value`` was a dict or valid JSON that decodes
        to an object; an empty dict otherwise (covers ``None``, blank
        strings, or non-object JSON), so callers never need to handle a
        parse failure explicitly.

    Example:
        >>> _parse_json_object('{"city": "Paris"}')
        {'city': 'Paris'}
        >>> _parse_json_object({"already": "a dict"})
        {'already': 'a dict'}
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

    The inverse of :func:`_parse_json_object`: Anthropic's ``tool_use.input``
    arrives as a parsed JSON object, but
    :class:`~src.providers.base.ToolCall.arguments` must be a string (see
    that field's docstring), so it's re-serialized here.

    Args:
        value: The value to serialize — typically the parsed ``input``
            object from an Anthropic ``tool_use`` block, but passed through
            unchanged if already a string (avoiding double-encoding).

    Returns:
        ``value`` unchanged if it's already a string; otherwise its JSON
        serialization.

    Example:
        >>> _serialize_json({"city": "Paris"})
        '{"city": "Paris"}'
        >>> _serialize_json("already a string")
        'already a string'
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
        >>> _optional_str("end_turn")
        'end_turn'
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
register_provider(AnthropicProvider)
