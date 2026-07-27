"""Unit tests for retry_async."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from src.providers.http_errors import HTTPErrorMapper
from src.providers.retry import retry_async


@pytest.fixture
def mapper() -> HTTPErrorMapper:
    return HTTPErrorMapper("test-provider")


@pytest.mark.asyncio
async def test_retry_async_succeeds_on_first_attempt(mapper: HTTPErrorMapper) -> None:
    operation = AsyncMock(return_value="ok")

    result = await retry_async(operation, mapper=mapper, max_retries=2)

    assert result == "ok"
    operation.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_async_retries_on_429(mapper: HTTPErrorMapper, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.providers.retry.asyncio.sleep", fake_sleep)

    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    http_error = httpx.HTTPStatusError("rate limited", request=request, response=response)
    success_response = httpx.Response(200, request=request)

    call_count = 0

    async def operation() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise http_error
        return success_response

    result = await retry_async(
        operation,
        mapper=mapper,
        max_retries=2,
        base_delay_seconds=0.01,
    )

    assert result.status_code == 200
    assert call_count == 2
    assert sleeps == [0.01]
