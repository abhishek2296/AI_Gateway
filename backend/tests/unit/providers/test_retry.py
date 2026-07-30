"""
Unit tests for `retry_async` (`src/providers/retry.py`).

`retry_async` wraps a provider HTTP call with exponential-backoff retries
for transient failures (e.g. HTTP 429/502/503/504, connection errors), using
an `HTTPErrorMapper` to decide which statuses are worth retrying. These
tests verify both the "no retry needed" happy path and that a retryable
failure is retried with the expected delay, without ever sleeping for real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from src.providers.http_errors import HTTPErrorMapper
from src.providers.retry import retry_async


@pytest.fixture
def mapper() -> HTTPErrorMapper:
    """Return an `HTTPErrorMapper` for a fake provider name, used to drive retry decisions."""
    return HTTPErrorMapper("test-provider")


@pytest.mark.asyncio
async def test_retry_async_succeeds_on_first_attempt(mapper: HTTPErrorMapper) -> None:
    """
    An operation that succeeds immediately is awaited exactly once, with no retry overhead.

    Confirms `retry_async` is a transparent pass-through when nothing goes
    wrong -- it must not introduce extra delay or duplicate calls for the
    common case where the underlying provider call just works.
    """
    operation = AsyncMock(return_value="ok")

    result = await retry_async(operation, mapper=mapper, max_retries=2)

    assert result == "ok"
    operation.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_async_retries_on_429(mapper: HTTPErrorMapper, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A single HTTP 429 (rate limited) is retried once and the retry ultimately succeeds.

    429 is one of the status codes `HTTPErrorMapper.should_retry_status`
    treats as transient/retryable, so callers shouldn't have to handle rate
    limiting themselves. `asyncio.sleep` is monkeypatched to a fake that
    just records the requested delay, so the test verifies the correct
    backoff delay was computed without actually waiting for it.
    """
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
        # Fails on the first call to simulate a transient 429, then succeeds
        # on the retry -- this is the scenario retry_async exists to handle.
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
    # base_delay_seconds=0.01 with no Retry-After header means the first
    # retry's exponential-backoff delay should be exactly the base delay.
    assert sleeps == [0.01]
