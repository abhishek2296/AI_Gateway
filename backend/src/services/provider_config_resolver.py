"""Translate resolved provider selection into ProviderFactory constructor kwargs."""

from __future__ import annotations

import logging
import os
from typing import Any

from src.core.config import Settings
from src.services.resolution_types import ResolvedProviderSelection

logger = logging.getLogger(__name__)


class ProviderConfigResolver:
    """
    Build provider-specific constructor kwargs from database configuration.

    Keeps provider parameter names out of :class:`~src.services.ai_service.AIService`.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        """Return kwargs for :meth:`~src.providers.factory.ProviderFactory.create`."""
        builder = self._builders.get(selection.provider_name, self._build_generic)
        kwargs = builder(selection)
        if kwargs.get("api_key"):
            logger.debug(
                "Resolved API key for provider=%s (identifier=%s)",
                selection.provider_name,
                selection.api_key.key_identifier if selection.api_key else "config",
            )
        return kwargs

    def _build_ollama(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        return {
            "base_url": self._base_url(selection, fallback=self._settings.OLLAMA_HOST),
            "timeout": self._timeout_seconds(selection),
        }

    def _build_openai(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": self._timeout_seconds(selection),
        }
        base_url = self._base_url(selection, fallback=None)
        if base_url:
            kwargs["base_url"] = base_url
        api_key = self._read_api_key(selection)
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    def _build_anthropic(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": self._timeout_seconds(selection),
            "api_version": self._settings.ANTHROPIC_API_VERSION,
        }
        base_url = self._base_url(selection, fallback=None)
        if base_url:
            kwargs["base_url"] = base_url
        api_key = self._read_api_key(selection)
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    def _build_gemini(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": self._timeout_seconds(selection),
        }
        base_url = self._base_url(selection, fallback=None)
        if base_url:
            kwargs["base_url"] = base_url
        api_key = self._read_api_key(selection)
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    def _build_generic(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": self._timeout_seconds(selection),
        }
        base_url = self._base_url(selection, fallback=None)
        if base_url:
            kwargs["base_url"] = base_url
        api_key = self._read_api_key(selection)
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    def _timeout_seconds(self, selection: ResolvedProviderSelection) -> float:
        if selection.configuration and selection.configuration.timeout_seconds is not None:
            return float(selection.configuration.timeout_seconds)
        return float(self._settings.TIMEOUT)

    def _base_url(
        self,
        selection: ResolvedProviderSelection,
        *,
        fallback: str | None,
    ) -> str | None:
        if selection.configuration and selection.configuration.endpoint:
            return selection.configuration.endpoint
        if selection.provider and selection.provider.base_url:
            return selection.provider.base_url
        return fallback

    def _read_api_key(self, selection: ResolvedProviderSelection) -> str | None:
        env_names: list[str] = []
        if selection.api_key and selection.api_key.api_key_env:
            env_names.append(selection.api_key.api_key_env)
        if selection.configuration and selection.configuration.api_key_env:
            env_names.append(selection.configuration.api_key_env)
        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                return value
        return None

    @property
    def _builders(self) -> dict[str, Any]:
        return {
            "ollama": self._build_ollama,
            "openai": self._build_openai,
            "anthropic": self._build_anthropic,
            "gemini": self._build_gemini,
        }
