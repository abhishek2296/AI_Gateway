"""
Unit tests for `HTTPErrorMapper` (`src/providers/http_errors.py`).

`HTTPErrorMapper` centralizes the translation from raw `httpx` exceptions
(HTTP status errors and transport-level request errors) into the gateway's
own `ProviderError` hierarchy, plus retry policy decisions (which statuses
are transient, and how long to wait based on a `Retry-After` header). These
tests verify that mapping and policy logic in isolation, independent of any
specific provider's wire format.
"""

from __future__ import annotations

import httpx
import pytest

from src.providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from src.providers.http_errors import HTTPErrorMapper


@pytest.fixture
def mapper() -> HTTPErrorMapper:
    """Return an `HTTPErrorMapper` for a fake provider name with a custom unavailable message."""
    return HTTPErrorMapper("test-provider", unavailable_message="Provider down.")


@pytest.mark.parametrize(
    ("status", "expected_type"),
    [
        (404, ModelNotFoundError),
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
        (502, ProviderUnavailableError),
        (418, ProviderError),
    ],
)
def test_map_http_error_status_codes(
    mapper: HTTPErrorMapper,
    status: int,
    expected_type: type[ProviderError],
) -> None:
    """
    Each HTTP status code is mapped to the correct `ProviderError` subclass.

    Covers every status branch in `HTTPErrorMapper.map_http_error`: 404 ->
    model not found, 400 -> invalid request, 401/403 -> both map to the same
    authentication error (a provider shouldn't need to distinguish
    "unauthenticated" from "forbidden" for callers), 429 -> rate limit, 5xx
    -> provider unavailable, and an unrecognized status (418, chosen as a
    stand-in for "some other 4xx/5xx we don't special-case") falls back to
    the generic `ProviderError`. This lets callers catch one exception type
    per failure category regardless of which vendor's status code it came
    from.
    """
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    mapped = mapper.map_http_error(exc)

    assert isinstance(mapped, expected_type)
    assert mapped.provider == "test-provider"


def test_map_request_error(mapper: HTTPErrorMapper) -> None:
    """
    A transport-level failure (e.g. connection refused) maps to `ProviderUnavailableError`.

    Unlike `map_http_error`, `map_request_error` handles the case where no
    HTTP response was ever received at all (DNS failure, connection reset,
    etc.) -- these should also surface as "provider unavailable" rather than
    leaking a raw `httpx.ConnectError` to callers.
    """
    request = httpx.Request("GET", "https://example.com")
    exc = httpx.ConnectError("connection failed", request=request)

    mapped = mapper.map_request_error(exc)

    assert isinstance(mapped, ProviderUnavailableError)
    assert mapped.provider == "test-provider"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, True), (502, True), (503, True), (504, True), (400, False), (500, False)],
)
def test_should_retry_status(mapper: HTTPErrorMapper, status: int, expected: bool) -> None:
    """
    Only 429/502/503/504 are considered retryable; other statuses (including plain 500) are not.

    This is the policy `retry_async` (`src/providers/retry.py`) consults
    before retrying. Plain 500 is deliberately excluded because a generic
    server error is less likely to be transient than a 502/503/504
    (gateway/overload) or 429 (rate limit) -- retrying it blindly could mask
    a real bug and waste time.
    """
    assert mapper.should_retry_status(status) is expected


def test_retry_after_seconds(mapper: HTTPErrorMapper) -> None:
    """
    A numeric `Retry-After` header value is parsed into a float number of seconds.

    Honoring the server-specified `Retry-After` (rather than always using
    our own fixed backoff schedule) lets the gateway respect a vendor's
    explicit rate-limit guidance when it's provided.
    """
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, headers={"Retry-After": "2.5"}, request=request)

    assert mapper.retry_after_seconds(response) == 2.5
