"""
Ollama REST adapter implementing :class:`~src.providers.base.BaseProvider`.

Ollama is the gateway's local/self-hosted provider: it runs on the same
machine or LAN (typically ``http://localhost:11434``), requires no API key,
and exposes a small REST API (``/api/chat``, ``/api/embeddings``,
``/api/tags``) with its own response shapes distinct from the OpenAI-style
conventions the cloud vendors loosely share. This module owns every bit of
that translation: building Ollama-shaped request payloads from the
normalized :class:`~src.providers.base.ChatRequest`/etc. types, and mapping
Ollama's JSON responses back into the normalized
:class:`~src.providers.base.ChatResponse`/etc. types.

Unlike the cloud adapters (``openai.py``, ``anthropic.py``, ``gemini.py``),
this adapter does not use :func:`~src.providers.retry.retry_async` — a local
Ollama instance either responds or is down/misconfigured, and blindly
retrying a request against a backend that isn't a shared, rate-limited,
occasionally-flaky cloud service is unlikely to help the way it can for
those vendors.
"""

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

    Example:
        >>> import asyncio
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> provider = OllamaProvider(base_url="http://localhost:11434")
        >>> request = ChatRequest(
        ...     model="qwen3:8b",
        ...     messages=[ChatMessage(role="user", content="Hello!")],
        ... )
        >>> response = asyncio.run(provider.chat(request))  # doctest: +SKIP
        >>> asyncio.run(provider.close())
    """

    provider_name = _PROVIDER

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Create an Ollama adapter targeting a specific server instance.

        Args:
            base_url: The base URL of the Ollama server, e.g.
                ``"http://localhost:11434"``. No credentials are required
                since Ollama has no built-in authentication.
            timeout: Request timeout in seconds for the internally created
                HTTP client, applied only when ``http_client`` is ``None``.
                Defaults to ``60.0`` seconds — generous relative to typical
                cloud-vendor defaults, since local model inference
                (especially on CPU or with large models) can legitimately
                take longer than a cloud API round-trip.
            http_client: An optional pre-built ``httpx.AsyncClient`` to reuse
                instead of constructing a new one (see
                :meth:`~src.providers.http_mixin.HTTPProviderMixin._init_http_client`
                for the ownership implications).

        Raises:
            (none directly) — delegates client setup to
            :meth:`~src.providers.http_mixin.HTTPProviderMixin._init_http_client`,
            which does not itself raise for these arguments.
        """
        self._init_http_client(
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            provider_label="OllamaProvider",
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a non-streaming chat completion via ``POST /api/chat``.

        Args:
            request: The normalized chat request to execute.

        Returns:
            The normalized chat completion response, mapped from Ollama's
            JSON body via :func:`_map_chat_response`.

        Raises:
            ModelNotFoundError: If Ollama returns HTTP 404 (e.g. the model
                hasn't been pulled locally).
            InvalidRequestError: If Ollama returns HTTP 400.
            AuthenticationError: If Ollama returns HTTP 401/403 (not expected
                in typical local deployments, but mapped for completeness if
                a proxy in front of Ollama enforces auth).
            RateLimitError: If Ollama returns HTTP 429.
            ProviderUnavailableError: If Ollama returns a 5xx response, or if
                the request fails at the transport level (connection
                refused, timeout, etc. — i.e. Ollama isn't running/reachable).
            ProviderError: For any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import ChatMessage, ChatRequest
            >>> provider = OllamaProvider(base_url="http://localhost:11434")
            >>> request = ChatRequest(
            ...     model="qwen3:8b",
            ...     messages=[ChatMessage(role="user", content="Hi")],
            ... )
            >>> asyncio.run(provider.chat(request))  # doctest: +SKIP
            ChatResponse(content='Hello!', model='qwen3:8b', ...)
        """
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
        """
        Stream chat completion chunks via ``POST /api/chat`` with ``stream=true``.

        Unlike the cloud adapters, Ollama does not use Server-Sent Events for
        streaming — it responds with newline-delimited JSON objects (one per
        line), each carrying an incremental ``message.content`` delta and a
        ``done`` boolean marking the final chunk. This method therefore
        parses lines directly with ``json.loads`` rather than using one of
        the shared ``streaming.py`` SSE/NDJSON helpers (which are consumed
        only by the cloud adapters).

        Args:
            request: The normalized chat request to execute, typically (but
                not necessarily) with ``request.stream`` set to ``True``.

        Yields:
            :class:`~src.providers.base.ChatStreamChunk` objects as each line
            of the streamed response arrives. The final chunk (where
            Ollama's ``done`` field is ``true``) carries ``finish_reason``
            and ``usage``; earlier chunks carry only ``content``.

        Raises:
            ProviderError: If Ollama emits a line that isn't valid JSON
                mid-stream — this should not happen under normal operation,
                but is caught explicitly (rather than left to propagate as a
                raw ``json.JSONDecodeError``) so callers only ever have to
                handle the provider exception hierarchy.
            ModelNotFoundError: If the initial request is rejected with HTTP
                404 (e.g. an invalid model name) before streaming begins.
            InvalidRequestError: If rejected with HTTP 400.
            AuthenticationError: If rejected with HTTP 401/403.
            RateLimitError: If rejected with HTTP 429.
            ProviderUnavailableError: If the request fails at the transport
                level, or Ollama returns a 5xx response, at any point before
                or during streaming.
            ProviderError: For any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import ChatMessage, ChatRequest
            >>> async def demo():
            ...     provider = OllamaProvider(base_url="http://localhost:11434")
            ...     request = ChatRequest(
            ...         model="qwen3:8b",
            ...         messages=[ChatMessage(role="user", content="Hi")],
            ...         stream=True,
            ...     )
            ...     async for chunk in provider.stream_chat(request):
            ...         print(chunk.content, end="")
            >>> asyncio.run(demo())  # doctest: +SKIP
        """
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
        """
        Generate embeddings via ``POST /api/embeddings``.

        Args:
            request: The normalized embeddings request to execute.

        Returns:
            The normalized embeddings response, mapped via
            :func:`_map_embeddings_response`.

        Raises:
            ModelNotFoundError: If Ollama returns HTTP 404 (e.g. the
                embedding model hasn't been pulled locally).
            InvalidRequestError: If Ollama returns HTTP 400.
            AuthenticationError: If Ollama returns HTTP 401/403.
            RateLimitError: If Ollama returns HTTP 429.
            ProviderUnavailableError: If Ollama returns a 5xx response or the
                request fails at the transport level.
            ProviderError: For any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> from src.providers.base import EmbeddingsRequest
            >>> provider = OllamaProvider(base_url="http://localhost:11434")
            >>> request = EmbeddingsRequest(model="nomic-embed-text", input="hello")
            >>> asyncio.run(provider.embeddings(request))  # doctest: +SKIP
            EmbeddingsResponse(model='nomic-embed-text', embeddings=[[...]], usage=None)
        """
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
        """
        List available models via ``GET /api/tags``.

        Unlike the cloud adapters, this method does not cache its result
        (no ``ModelListCache``): Ollama's model list reflects whatever is
        locally installed, which can change at any time (a user pulling or
        removing a model), and — being a local call — refetching it is cheap
        enough that caching isn't worth the staleness risk.

        Returns:
            A tuple of :class:`~src.providers.base.ModelInfo`, one per model
            Ollama reports as installed, mapped via :func:`_map_models`.

        Raises:
            AuthenticationError: If Ollama returns HTTP 401/403.
            RateLimitError: If Ollama returns HTTP 429.
            ProviderUnavailableError: If Ollama returns a 5xx response or the
                request fails at the transport level.
            ProviderError: For any other unmapped non-2xx status code.

        Example:
            >>> import asyncio
            >>> provider = OllamaProvider(base_url="http://localhost:11434")
            >>> asyncio.run(provider.list_models())  # doctest: +SKIP
            (ModelInfo(id='qwen3:8b', name='qwen3:8b', ...),)
        """
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

        Reuses the ``/api/tags`` endpoint (the same one :meth:`list_models`
        calls) rather than a dedicated health endpoint, since Ollama doesn't
        expose one — successfully listing tags is itself sufficient proof
        the server is up and responding to API calls.

        Returns:
            A :class:`~src.providers.base.HealthCheckResult` with
            ``healthy=True`` and the installed model count when the probe
            succeeds; ``healthy=False`` with a diagnostic ``message``/
            ``details`` when the transport fails or Ollama returns a
            non-2xx status.

        Raises:
            (none) — by design, this method never propagates an exception;
            every failure path is caught and encoded into the returned
            result instead (see :meth:`~src.providers.base.BaseProvider.health_check`).

        Example:
            >>> import asyncio
            >>> provider = OllamaProvider(base_url="http://localhost:11434")
            >>> asyncio.run(provider.health_check())  # doctest: +SKIP
            HealthCheckResult(healthy=True, ...)
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
    """
    Build the JSON body for Ollama's ``POST /api/chat`` endpoint.

    Ollama nests sampling parameters (temperature, max token count) under an
    ``options`` object rather than as top-level request fields, and uses
    ``num_predict`` (not ``max_tokens``) as its name for the output token
    cap — both vendor-specific quirks handled here so the rest of the
    codebase never needs to know Ollama's particular payload shape.

    Args:
        request: The normalized chat request to translate.
        stream: Whether this payload is for a streaming (``stream_chat``) or
            non-streaming (``chat``) call. Keyword-only to make call sites
            self-documenting (``_build_chat_payload(request, stream=True)``
            reads unambiguously, whereas a positional bool would not).

    Returns:
        A JSON-serializable dict matching Ollama's ``/api/chat`` request
        schema.

    Example:
        >>> from src.providers.base import ChatMessage, ChatRequest
        >>> request = ChatRequest(
        ...     model="qwen3:8b",
        ...     messages=[ChatMessage(role="user", content="Hi")],
        ...     temperature=0.5,
        ...     max_tokens=100,
        ... )
        >>> _build_chat_payload(request, stream=False)
        {'model': 'qwen3:8b', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': False, 'options': {'temperature': 0.5, 'num_predict': 100}}
    """
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
    """
    Extract token usage from an Ollama response body, if present.

    Ollama names its usage fields ``prompt_eval_count``/``eval_count``
    rather than the ``prompt_tokens``/``completion_tokens`` convention used
    by OpenAI/Anthropic — normalized here into the shared
    :class:`~src.providers.base.TokenUsage` shape. Ollama also does not
    report a combined total directly, so it is computed here as the sum of
    the two counts (only when both are present, to avoid fabricating a
    partial total from just one side).

    Args:
        data: The raw JSON response body (or a streaming chunk) from Ollama,
            as a mapping.

    Returns:
        A :class:`~src.providers.base.TokenUsage` if either count was
        present in ``data``; ``None`` if neither was reported (e.g. an
        intermediate streaming chunk, which only carries usage data once
        ``done`` is ``true``).

    Example:
        >>> _map_token_usage({"prompt_eval_count": 10, "eval_count": 5})
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        >>> _map_token_usage({}) is None
        True
    """
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
    """
    Map an Ollama ``/api/chat`` (non-streaming) response body into a :class:`ChatResponse`.

    Args:
        data: The parsed JSON response body from Ollama.

    Returns:
        The normalized :class:`~src.providers.base.ChatResponse`. Ollama has
        no concept of tool calls or a provider-assigned response id in its
        response schema, so those fields are left at their defaults
        (``None``).

    Example:
        >>> _map_chat_response({
        ...     "model": "qwen3:8b",
        ...     "message": {"content": "Hi there!"},
        ...     "done_reason": "stop",
        ...     "prompt_eval_count": 5,
        ...     "eval_count": 3,
        ... })
        ChatResponse(content='Hi there!', model='qwen3:8b', finish_reason='stop', usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8), provider_response_id=None, tool_calls=None, details={})
    """
    message = data.get("message") or {}
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    return ChatResponse(
        content=str(content),
        model=str(data.get("model", "")),
        finish_reason=_optional_str(data.get("done_reason")),
        usage=_map_token_usage(data),
    )


def _map_stream_chunk(data: Mapping[str, Any]) -> ChatStreamChunk:
    """
    Map one line of an Ollama streaming response into a :class:`ChatStreamChunk`.

    Ollama's ``done`` boolean marks the final chunk of the stream, at which
    point (and only then) it also populates ``done_reason`` and the usage
    count fields — earlier chunks omit those entirely. This function mirrors
    that by only reading ``finish_reason``/``usage`` when ``done`` is
    ``true``, so intermediate chunks aren't misreported as carrying (empty)
    finish/usage data.

    Args:
        data: One parsed JSON line from the streaming response body.

    Returns:
        The normalized :class:`~src.providers.base.ChatStreamChunk` for this
        line.

    Example:
        >>> _map_stream_chunk({"message": {"content": "Hi"}, "done": False})
        ChatStreamChunk(content='Hi', finish_reason=None, usage=None, tool_calls=None)
        >>> _map_stream_chunk({
        ...     "message": {"content": ""},
        ...     "done": True,
        ...     "done_reason": "stop",
        ...     "prompt_eval_count": 5,
        ...     "eval_count": 3,
        ... })
        ChatStreamChunk(content='', finish_reason='stop', usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8), tool_calls=None)
    """
    message = data.get("message") or {}
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    done = bool(data.get("done"))
    return ChatStreamChunk(
        content=str(content),
        finish_reason=_optional_str(data.get("done_reason")) if done else None,
        usage=_map_token_usage(data) if done else None,
    )


def _map_embeddings_response(data: Mapping[str, Any]) -> EmbeddingsResponse:
    """
    Map an Ollama ``/api/embeddings`` response body into an :class:`EmbeddingsResponse`.

    Args:
        data: The parsed JSON response body from Ollama.

    Returns:
        The normalized :class:`~src.providers.base.EmbeddingsResponse`.
        Malformed or missing embedding entries are silently skipped (rather
        than raising) so a single unexpected entry doesn't fail the whole
        response when the rest of the payload is usable.

    Example:
        >>> _map_embeddings_response({"model": "nomic-embed-text", "embeddings": [[0.1, 0.2]]})
        EmbeddingsResponse(model='nomic-embed-text', embeddings=[[0.1, 0.2]], usage=None)
    """
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
    """
    Map an Ollama ``/api/tags`` response body into a tuple of :class:`ModelInfo`.

    Ollama's ``/api/tags`` payload provides no explicit capability flags at
    all, so every ``supports_*`` field on the resulting
    :class:`~src.providers.base.ModelInfo` is a heuristic:

    - ``supports_streaming=True`` unconditionally — every Ollama model
      supports streaming via the same ``/api/chat`` endpoint.
    - ``supports_vision=True`` unconditionally — this is a known
      simplification; not every locally installed model is actually
      multimodal, but Ollama's tags response doesn't distinguish
      vision-capable models, so this errs on the permissive side rather
      than blocking image input outright.
    - ``supports_tools=True`` unconditionally, for the same reason.
    - ``supports_embeddings`` is inferred from whether the substring
      ``"embed"`` appears in the model's name (e.g. ``"nomic-embed-text"``),
      since embedding-only models are conventionally named that way.
    - ``supports_json=False`` unconditionally, since Ollama's ``/api/chat``
      JSON-mode support isn't reflected in this endpoint either.

    Args:
        data: The parsed JSON response body from ``GET /api/tags``.

    Returns:
        A tuple of :class:`~src.providers.base.ModelInfo`. Entries missing a
        usable name are skipped; if ``data["models"]`` isn't a list at all,
        an empty tuple is returned rather than raising.

    Example:
        >>> _map_models({"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]})
        (ModelInfo(id='qwen3:8b', name='qwen3:8b', display_name='qwen3:8b', context_window=None, supports_streaming=True, supports_embeddings=False, supports_vision=True, supports_tools=True, supports_json=False),)
    """
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
    """
    Coerce a possibly-``None`` value to ``str``, preserving ``None``.

    A small shared helper so response-mapping functions can write
    ``_optional_str(data.get("done_reason"))`` instead of repeating a
    ``None``-check/``str()`` pattern inline at every call site.

    Args:
        value: Any value, typically from a parsed JSON response.

    Returns:
        ``None`` if ``value`` is ``None``; otherwise ``str(value)``.

    Example:
        >>> _optional_str(None) is None
        True
        >>> _optional_str("stop")
        'stop'
        >>> _optional_str(42)
        '42'
    """
    if value is None:
        return None
    return str(value)


def _as_int(value: object) -> int | None:
    """
    Best-effort coercion of a JSON value to ``int``, tolerating bad input.

    Vendor JSON responses are not guaranteed to be well-typed (a field that
    is normally a number could theoretically arrive as a string or be
    missing); this helper treats any value that can't cleanly become an
    ``int`` as simply absent (``None``) rather than raising and failing the
    whole response mapping over one malformed field.

    Args:
        value: Any value, typically from a parsed JSON response.

    Returns:
        The integer value, or ``None`` if ``value`` is ``None`` or cannot be
        converted to ``int``.

    Example:
        >>> _as_int(5)
        5
        >>> _as_int("7")
        7
        >>> _as_int(None) is None
        True
        >>> _as_int("not-a-number") is None
        True
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Registering at import time (rather than requiring an explicit call from
# application startup code) is what lets `import src.providers` alone make
# this adapter discoverable via the registry — see `src/providers/__init__.py`.
register_provider(OllamaProvider)
