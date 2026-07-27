"""Authentication header builders for HTTP provider adapters."""

from __future__ import annotations

from typing import Any


def openai_headers(
    api_key: str,
    *,
    organization: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization:
        headers["OpenAI-Organization"] = organization
    if extra:
        headers.update(extra)
    return headers


def anthropic_headers(
    api_key: str,
    *,
    api_version: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": api_version,
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def gemini_headers(
    api_key: str,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def merge_headers(base: dict[str, str], extra: dict[str, Any] | None) -> dict[str, str]:
    if not extra:
        return base
    merged = dict(base)
    for key, value in extra.items():
        if value is not None:
            merged[str(key)] = str(value)
    return merged
