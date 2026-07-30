"""
Unit tests for streaming helpers (`src/providers/streaming.py`).

These low-level async generators parse the raw line-based streaming formats
used by provider APIs -- plain Server-Sent Events (SSE), SSE with paired
``event:``/``data:`` lines (Anthropic-style), and newline-delimited JSON
(Ollama-style) -- into normalized Python dicts/strings. Testing them
directly (rather than only through a full provider's `stream_chat`) isolates
parsing edge cases like the `[DONE]` sentinel and blank-line handling from
any single vendor's response schema.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.providers.streaming import iter_ndjson, iter_sse_data_lines, iter_sse_events, iter_sse_json


@pytest.mark.asyncio
async def test_iter_sse_data_lines_skips_done_marker() -> None:
    """
    Raw SSE ``data:`` payloads are yielded as strings, and the ``[DONE]`` sentinel is dropped.

    OpenAI-style SSE streams end with a literal ``data: [DONE]`` line that is
    not JSON and carries no payload -- callers should never see it, since
    forwarding it downstream would either be ignored noise or crash a JSON
    parser expecting a real payload.
    """
    body = 'data: {"a": 1}\n\ndata: [DONE]\n\ndata: {"b": 2}\n'
    response = httpx.Response(200, content=body.encode())

    payloads = [payload async for payload in iter_sse_data_lines(response)]

    assert payloads == ['{"a": 1}', '{"b": 2}']


@pytest.mark.asyncio
async def test_iter_sse_json_parses_objects() -> None:
    """`iter_sse_json` parses each SSE ``data:`` payload as a JSON object, not just a raw string."""
    body = 'data: {"content":"hi"}\n'
    response = httpx.Response(200, content=body.encode())

    chunks = [chunk async for chunk in iter_sse_json(response)]

    assert chunks == [{"content": "hi"}]


@pytest.mark.asyncio
async def test_iter_ndjson_parses_lines() -> None:
    """
    `iter_ndjson` parses one JSON object per non-blank line and skips blank lines.

    Ollama's streaming responses are newline-delimited JSON (not SSE), so
    this helper has no ``data:`` prefix to strip; the blank line in the test
    body confirms it's tolerated rather than raising a JSON parse error.
    """
    body = '{"x":1}\n\n{"x":2}\n'
    response = httpx.Response(200, content=body.encode())

    chunks = [chunk async for chunk in iter_ndjson(response)]

    assert chunks == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_iter_sse_events_pairs_event_and_data() -> None:
    """
    `iter_sse_events` associates each ``data:`` payload with the preceding ``event:`` name.

    Anthropic's streaming API sends a named ``event:`` line immediately
    before its ``data:`` payload (e.g. ``content_block_delta``,
    ``message_delta``), and the event name determines how the payload should
    be interpreted. This confirms the event/data pairing survives across
    multiple SSE frames, not just the first one.
    """
    body = (
        "event: content_block_delta\n"
        f"data: {json.dumps({'delta': {'text': 'Hi'}})}\n\n"
        "event: message_delta\n"
        f"data: {json.dumps({'delta': {'stop_reason': 'end_turn'}})}\n"
    )
    response = httpx.Response(200, content=body.encode())

    events = [event async for event in iter_sse_events(response)]

    assert events[0][0] == "content_block_delta"
    assert events[0][1]["delta"]["text"] == "Hi"
    assert events[1][0] == "message_delta"
