"""
SSE and streaming helpers for provider HTTP adapters.

Cloud vendors expose streaming completions over HTTP using a handful of
different wire formats:

- OpenAI and (in SSE mode) Gemini use classic `Server-Sent Events
  <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events>`_:
  newline-delimited ``data: <json>`` lines terminated by a special
  ``data: [DONE]`` sentinel.
- Anthropic's Messages API also uses SSE, but additionally pairs each
  ``data:`` payload with a preceding ``event: <type>`` line that names the
  event (e.g. ``content_block_delta``, ``message_delta``) so the client knows
  how to interpret the JSON without inspecting its shape.
- Gemini's ``streamGenerateContent`` endpoint can *also* respond with plain
  newline-delimited JSON (NDJSON) instead of SSE, depending on how the
  request is made; the adapter picks the right parser based on the
  response's ``Content-Type`` header (see ``gemini.py``).

This module implements one small, focused async generator per wire format so
each vendor adapter's ``stream_chat`` method can simply iterate over parsed
JSON objects (or ``(event_type, json)`` pairs) instead of re-implementing
line-buffering and payload-stripping logic per provider.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx


async def iter_sse_data_lines(response: httpx.Response) -> AsyncIterator[str]:
    """
    Yield non-empty SSE ``data`` payload lines from a streaming response.

    Server-Sent Events frame each message as a line beginning with
    ``data:`` (with everything before/after governed by the SSE spec, e.g.
    optional blank "event separator" lines). This generator strips the
    ``data:`` prefix and surrounding whitespace, skips blank payloads (which
    SSE implementations sometimes emit as keep-alive pings), and also skips
    the vendor-standard ``[DONE]`` sentinel line that OpenAI (and others)
    send to mark the end of the stream — since it isn't JSON and callers
    only care about actual content payloads.

    Args:
        response: An already-open streaming ``httpx.Response`` (i.e. obtained
            via ``client.stream(...)`` and used as an async context manager)
            whose body is being read incrementally line by line.

    Returns:
        An async iterator of raw payload strings (the text following
        ``data:``), each expected to be a JSON-encoded object but returned
        here as-is, before any parsing.

    Raises:
        httpx.StreamError: If the underlying connection is interrupted while
            reading lines (propagated from ``response.aiter_lines()``, not
            raised directly by this function).

    Example:
        Given a response body of::

            data: {"content": "Hello"}

            data: {"content": " world"}

            data: [DONE]

        iterating yields ``'{"content": "Hello"}'`` then
        ``'{"content": " world"}'``, with the blank keep-alive line and the
        ``[DONE]`` sentinel both skipped.
    """
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        yield payload


async def iter_sse_json(response: httpx.Response) -> AsyncIterator[Mapping[str, Any]]:
    """
    Parse SSE ``data`` lines as JSON objects.

    Thin layer over :func:`iter_sse_data_lines` that additionally
    JSON-decodes each payload. Used by ``OpenAIProvider.stream_chat`` (whose
    payloads are always plain JSON objects with no accompanying ``event:``
    type) and by ``GeminiProvider.stream_chat`` when Gemini responds in SSE
    mode.

    Args:
        response: An already-open streaming ``httpx.Response`` being read via
            SSE.

    Returns:
        An async iterator of parsed JSON objects (as ``Mapping[str, Any]``).
        Payloads that decode to a JSON value other than an object (e.g. a
        bare list or number) are silently skipped rather than yielded or
        raising, since every provider using this helper always sends objects
        for genuine content chunks.

    Raises:
        json.JSONDecodeError: If a ``data:`` payload is not valid JSON at
            all (as opposed to valid JSON of the wrong type, which is
            silently skipped).

    Example:
        Given SSE lines ``data: {"choices": [...]}"``, iterating yields the
        parsed ``dict`` for each chunk; a stray ``data: 42`` line (valid JSON
        but not an object) would be skipped rather than yielded.
    """
    async for payload in iter_sse_data_lines(response):
        data = json.loads(payload)
        if isinstance(data, dict):
            yield data


async def iter_ndjson(response: httpx.Response) -> AsyncIterator[Mapping[str, Any]]:
    """
    Parse newline-delimited JSON objects from a streaming response.

    Used as the fallback parser for ``GeminiProvider.stream_chat`` when
    Gemini's ``streamGenerateContent`` endpoint responds with plain NDJSON
    (one JSON object per line) rather than SSE — which endpoint/request
    parameter combination Gemini uses depends on the ``alt`` query parameter
    and isn't fully within the adapter's control, so both formats are
    supported and selected based on the response's ``Content-Type`` header.

    Args:
        response: An already-open streaming ``httpx.Response`` whose body is
            plain NDJSON (no ``data:``/``event:`` framing).

    Returns:
        An async iterator of parsed JSON objects. Blank lines are skipped
        (some NDJSON producers emit them between records); lines that parse
        to a non-object JSON value are silently skipped, mirroring
        :func:`iter_sse_json`'s behavior.

    Raises:
        json.JSONDecodeError: If a non-blank line is not valid JSON.

    Example:
        Given a response body of::

            {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}
            {"candidates": [{"content": {"parts": [{"text": "!"}]}}]}

        iterating yields the two parsed objects in order.
    """
    async for line in response.aiter_lines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            yield data


async def iter_sse_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str | None, Mapping[str, Any]]]:
    """
    Parse SSE streams that pair ``event:`` lines with ``data:`` JSON payloads.

    Anthropic's Messages API streaming format sends an ``event: <type>``
    line immediately before each ``data: <json>`` line (e.g.
    ``event: content_block_delta`` followed by
    ``data: {"delta": {"type": "text_delta", ...}}``). The event type tells
    the consumer how to interpret the JSON payload (see
    ``anthropic._map_stream_event``, which switches on exactly the event
    types this parser yields) without needing to guess from the JSON shape
    alone, since several event types share similar-looking payload
    structures.

    The parser tracks the most recently seen ``event:`` line and pairs it
    with the ``data:`` line that follows, then resets to ``None`` after
    yielding — this assumes (per the Anthropic SSE format) that ``event:``
    and ``data:`` always appear as an adjacent pair, so a stray ``data:``
    line with no preceding ``event:`` yields ``(None, data)`` rather than
    reusing a stale event type from an earlier pair.

    Args:
        response: An already-open streaming ``httpx.Response`` whose body
            follows Anthropic's ``event:``/``data:`` SSE convention.

    Returns:
        An async iterator of ``(event_type, data)`` tuples, where
        ``event_type`` is the string following ``event:`` (or ``None`` if a
        ``data:`` line arrived without one), and ``data`` is the parsed JSON
        object. The ``[DONE]`` sentinel and blank payloads are skipped, same
        as :func:`iter_sse_data_lines`. Non-object JSON payloads are also
        skipped.

    Raises:
        json.JSONDecodeError: If a ``data:`` payload is not valid JSON.

    Example:
        Given SSE lines::

            event: content_block_delta
            data: {"delta": {"type": "text_delta", "text": "Hi"}}

        iterating yields
        ``('content_block_delta', {'delta': {'type': 'text_delta', 'text': 'Hi'}})``.
    """
    event_type: str | None = None
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_type = line.removeprefix("event:").strip() or None
            continue
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        data = json.loads(payload)
        if isinstance(data, dict):
            yield event_type, data
        # Reset after each data line (paired or not) so a subsequent
        # unlabeled `data:` line doesn't incorrectly inherit this event's
        # type.
        event_type = None
