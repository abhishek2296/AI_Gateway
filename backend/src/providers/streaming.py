"""SSE and streaming helpers for provider HTTP adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx


async def iter_sse_data_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield non-empty SSE ``data`` payload lines from a streaming response."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        yield payload


async def iter_sse_json(response: httpx.Response) -> AsyncIterator[Mapping[str, Any]]:
    """Parse SSE ``data`` lines as JSON objects."""
    async for payload in iter_sse_data_lines(response):
        data = json.loads(payload)
        if isinstance(data, dict):
            yield data


async def iter_ndjson(response: httpx.Response) -> AsyncIterator[Mapping[str, Any]]:
    """Parse newline-delimited JSON objects from a streaming response."""
    async for line in response.aiter_lines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            yield data


async def iter_sse_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str | None, Mapping[str, Any]]]:
    """Parse SSE streams that pair ``event:`` lines with ``data:`` JSON payloads."""
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
        event_type = None
