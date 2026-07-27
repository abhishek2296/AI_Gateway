"""Unit tests for HTTPErrorMapper."""

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
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    mapped = mapper.map_http_error(exc)

    assert isinstance(mapped, expected_type)
    assert mapped.provider == "test-provider"


def test_map_request_error(mapper: HTTPErrorMapper) -> None:
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
    assert mapper.should_retry_status(status) is expected


def test_retry_after_seconds(mapper: HTTPErrorMapper) -> None:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, headers={"Retry-After": "2.5"}, request=request)

    assert mapper.retry_after_seconds(response) == 2.5
