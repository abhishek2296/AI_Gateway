"""Unit tests for streaming helpers."""

from __future__ import annotations

import json

import httpx
import pytest

from src.providers.streaming import iter_ndjson, iter_sse_data_lines, iter_sse_events, iter_sse_json


@pytest.mark.asyncio
async def test_iter_sse_data_lines_skips_done_marker() -> None:
    body = 'data: {"a": 1}\n\ndata: [DONE]\n\ndata: {"b": 2}\n'
    response = httpx.Response(200, content=body.encode())

    payloads = [payload async for payload in iter_sse_data_lines(response)]

    assert payloads == ['{"a": 1}', '{"b": 2}']


@pytest.mark.asyncio
async def test_iter_sse_json_parses_objects() -> None:
    body = 'data: {"content":"hi"}\n'
    response = httpx.Response(200, content=body.encode())

    chunks = [chunk async for chunk in iter_sse_json(response)]

    assert chunks == [{"content": "hi"}]


@pytest.mark.asyncio
async def test_iter_ndjson_parses_lines() -> None:
    body = '{"x":1}\n\n{"x":2}\n'
    response = httpx.Response(200, content=body.encode())

    chunks = [chunk async for chunk in iter_ndjson(response)]

    assert chunks == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_iter_sse_events_pairs_event_and_data() -> None:
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
