"""Model list caching for provider validation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.providers.base import ModelInfo


@dataclass(slots=True)
class _ModelCacheEntry:
    models: tuple[ModelInfo, ...]
    expires_at: float


class ModelListCache:
    """Short-lived in-memory cache of provider model catalogs."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _ModelCacheEntry] = {}

    def get(self, provider: str) -> tuple[ModelInfo, ...] | None:
        entry = self._entries.get(provider)
        if entry is None or entry.expires_at <= time.monotonic():
            return None
        return entry.models

    def set(self, provider: str, models: tuple[ModelInfo, ...]) -> None:
        self._entries[provider] = _ModelCacheEntry(
            models=models,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    def clear(self) -> None:
        self._entries.clear()
