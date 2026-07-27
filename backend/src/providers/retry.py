"""Async retry helper for provider HTTP calls."""

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

    Retries on transport errors and HTTP 429/502/503/504 responses.
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
