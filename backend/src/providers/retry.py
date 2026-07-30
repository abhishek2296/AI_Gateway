"""
Async retry helper for provider HTTP calls.

Cloud provider APIs (OpenAI, Anthropic, Gemini) occasionally fail with
transient errors — rate limiting, brief backend unavailability, or a dropped
connection — that typically succeed if retried a moment later.
:func:`retry_async` wraps a single HTTP call with bounded exponential
backoff so every HTTP-based adapter (``OpenAIProvider``, ``AnthropicProvider``,
``GeminiProvider``) gets the same retry behavior without duplicating the
backoff/logging logic in each adapter module. (``OllamaProvider`` does not
use this helper: a locally-hosted Ollama instance either responds or is down,
and retrying against a local backend after a network-style failure is
unlikely to help the way it can for a shared, load-balanced cloud API.)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from src.providers.http_errors import HTTPErrorMapper

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")


async def retry_async(
    operation: Callable[[], Awaitable[ResultT]],
    *,
    mapper: HTTPErrorMapper,
    max_retries: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
) -> ResultT:
    """
    Execute ``operation`` with exponential backoff on retryable HTTP failures.

    Retries on transport errors (``httpx.RequestError`` — connection drops,
    timeouts, DNS failures) and on HTTP responses whose status code
    ``mapper.should_retry_status`` considers retryable (429, 502, 503, 504 —
    see :meth:`~src.providers.http_errors.HTTPErrorMapper.should_retry_status`).
    All other failures (e.g. 400, 401, 404) propagate immediately on the
    first attempt since retrying them would just reproduce the same error.

    The backoff delay doubles with each attempt (``base_delay_seconds *
    2**attempt``), capped at ``max_delay_seconds`` — a standard "exponential
    backoff with a ceiling" strategy that avoids hammering a struggling
    backend with rapid-fire retries while still keeping the worst-case wait
    bounded and predictable. When the failure is a 429 response and the
    vendor supplied a ``Retry-After`` header, that explicit hint takes
    priority over the computed exponential delay (see
    :meth:`~src.providers.http_errors.HTTPErrorMapper.retry_after_seconds`)
    since the vendor knows its own rate-limit window better than a generic
    backoff formula could guess.

    ``operation`` is a zero-argument async callable (typically a small
    closure created by the caller, e.g. ``async def _call(): ...``) rather
    than accepting request arguments directly, so the same helper works for
    any HTTP verb/endpoint shape without needing a specialized signature per
    call site.

    Args:
        operation: A zero-argument async callable that performs one attempt
            of the underlying HTTP call and raises on failure (e.g. via
            ``response.raise_for_status()``). Called again, unmodified, on
            each retry attempt.
        mapper: The :class:`~src.providers.http_errors.HTTPErrorMapper`
            bound to the calling provider, used to decide which HTTP status
            codes are retryable and to read the ``Retry-After`` hint. Not
            used to build the exception raised by this function directly —
            callers are expected to catch the original
            ``httpx.HTTPStatusError``/``httpx.RequestError`` this function
            re-raises and map it themselves (see how ``OpenAIProvider.chat``
            calls ``retry_async`` and then ``self._mapper.map_http_error``).
        max_retries: Maximum number of *additional* attempts after the first
            one fails, e.g. ``max_retries=3`` means up to 4 total attempts.
            Defaults to ``3``, chosen as a balance between resilience to
            brief blips and not letting a single request hang indefinitely
            behind repeated retries.
        base_delay_seconds: The initial backoff delay before the first retry,
            in seconds. Defaults to ``0.5`` — long enough to give a
            momentary glitch a chance to clear, short enough not to add
            noticeable latency to a request that only needed one retry.
        max_delay_seconds: The maximum backoff delay any single retry will
            wait, in seconds, regardless of how many attempts have already
            been made. Defaults to ``8.0`` to keep the worst-case per-retry
            wait bounded even after several doublings.

    Returns:
        The result of ``operation()`` from whichever attempt first succeeds.

    Raises:
        httpx.HTTPStatusError: If every attempt fails with a retryable status
            and ``max_retries`` is exhausted, or immediately if the status is
            not retryable at all.
        httpx.RequestError: If every attempt fails with a transport error and
            ``max_retries`` is exhausted.

    Example:
        >>> import httpx
        >>> from src.providers.http_errors import HTTPErrorMapper
        >>> mapper = HTTPErrorMapper("openai")
        >>> attempts = {"count": 0}
        >>> async def flaky_call():
        ...     attempts["count"] += 1
        ...     if attempts["count"] < 2:
        ...         request = httpx.Request("GET", "https://api.openai.com/v1/models")
        ...         response = httpx.Response(503, request=request)
        ...         raise httpx.HTTPStatusError("busy", request=request, response=response)
        ...     return "ok"
        >>> import asyncio
        >>> asyncio.run(retry_async(flaky_call, mapper=mapper, base_delay_seconds=0.01))
        'ok'
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if attempt >= max_retries or not mapper.should_retry_status(status):
                raise
            delay = mapper.retry_after_seconds(exc.response) or min(
                base_delay_seconds * (2**attempt),
                max_delay_seconds,
            )
            logger.warning(
                "Retrying %s request after HTTP %d (attempt %d, delay=%.2fs)",
                # Accessing the mapper's "private" provider name is an
                # intentional, module-internal shortcut: retry.py and
                # http_errors.py are tightly coupled parts of the same HTTP
                # infrastructure layer, so this avoids adding a public
                # property purely for a log line.
                mapper._provider,
                status,
                attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
        except httpx.RequestError:
            if attempt >= max_retries:
                raise
            delay = min(base_delay_seconds * (2**attempt), max_delay_seconds)
            logger.warning(
                "Retrying %s request after transport error (attempt %d, delay=%.2fs)",
                mapper._provider,
                attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
