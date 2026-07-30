"""
Translate a resolved provider selection into ``ProviderFactory`` constructor kwargs.

This module is the "how do we connect to it?" half of provider resolution
(the "which provider/model?" half lives in ``provider_resolver.py``). Given a
:class:`~src.services.resolution_types.ResolvedProviderSelection` — provider
identity plus its raw database rows — :class:`ProviderConfigResolver` builds
the exact keyword arguments that
:meth:`~src.providers.factory.ProviderFactory.create` needs to instantiate a
concrete :class:`~src.providers.base.BaseProvider` adapter (base URL, timeout,
API key, etc).

Centralizing this translation here keeps provider-specific parameter names
(e.g. Anthropic's ``api_version``) out of
:class:`~src.services.ai_service.AIService`, which only ever needs to know
"call the factory with these kwargs" — not which kwargs a given provider
expects.
"""

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

    Dispatches to a per-provider "builder" method (``_build_ollama``,
    ``_build_openai``, etc.) keyed by provider name, falling back to
    ``_build_generic`` for any provider without bespoke handling. Each
    builder reads from three layers, in priority order, via the shared
    ``_base_url`` / ``_timeout_seconds`` / ``_read_api_key`` helpers:

    1. Database ``ProviderConfiguration`` (per-provider admin-configured
       overrides — endpoint, timeout, API key env var name).
    2. Database ``Provider`` row (e.g. ``base_url`` set at provider
       registration time).
    3. Application ``Settings`` / hardcoded default (used only when neither
       database source has a value).

    Keeping this dispatch and precedence logic here means
    :class:`~src.services.ai_service.AIService` never needs to know
    provider-specific parameter names.

    Example:
        >>> resolver = ProviderConfigResolver(get_settings())
        >>> kwargs = resolver.build(selection)
        >>> kwargs["timeout"]
        30.0
    """

    def __init__(self, settings: Settings) -> None:
        """
        Store the application settings used as the lowest-priority config source.

        Args:
            settings: Application settings providing fallback values (e.g.
                ``OLLAMA_HOST``, ``TIMEOUT``, ``ANTHROPIC_API_VERSION``) used
                when neither the database configuration nor the provider row
                supplies a value.
        """
        self._settings = settings

    def build(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        """
        Build the constructor kwargs for a resolved provider selection.

        Looks up the provider-specific builder by ``selection.provider_name``
        (falling back to the generic builder for unrecognized/future
        providers), then logs — at debug level, without leaking the secret
        value — whether an API key was successfully resolved.

        Args:
            selection: The resolved provider/model identity and its
                supporting database rows, as produced by
                :meth:`~src.services.provider_resolver.ProviderResolver.resolve`.

        Returns:
            A ``dict`` of keyword arguments ready to be unpacked into
            :meth:`~src.providers.factory.ProviderFactory.create`. Exact keys
            vary by provider (e.g. Ollama always includes ``base_url``, while
            OpenAI/Anthropic/Gemini only include it when configured).

        Example:
            >>> resolver.build(ollama_selection)
            {'base_url': 'http://localhost:11434', 'timeout': 30.0}
        """
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
        """
        Build kwargs for the local/self-hosted Ollama provider.

        Unlike the other builders, ``base_url`` is always included (Ollama
        has no concept of "no base URL" — it must always point somewhere),
        defaulting to ``Settings.OLLAMA_HOST`` when no database override
        exists. No API key is read since Ollama does not require one.

        Args:
            selection: The resolved provider selection for the ``ollama``
                provider.

        Returns:
            A ``dict`` with ``base_url`` and ``timeout`` keys.
        """
        return {
            "base_url": self._base_url(selection, fallback=self._settings.OLLAMA_HOST),
            "timeout": self._timeout_seconds(selection),
        }

    def _build_openai(self, selection: ResolvedProviderSelection) -> dict[str, Any]:
        """
        Build kwargs for the OpenAI provider.

        ``base_url`` and ``api_key`` are included only when a value is
        actually available (unlike Ollama, OpenAI's own SDK has sensible
        defaults, so we should not force an empty/placeholder value).

        Args:
            selection: The resolved provider selection for the ``openai``
                provider.

        Returns:
            A ``dict`` always containing ``timeout``, and conditionally
            ``base_url`` / ``api_key`` when resolved.
        """
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
        """
        Build kwargs for the Anthropic provider.

        Identical to the OpenAI builder except Anthropic additionally
        requires an ``api_version`` header value, which comes from
        ``Settings.ANTHROPIC_API_VERSION`` since it is a protocol-level
        constant rather than something configured per-provider in the
        database.

        Args:
            selection: The resolved provider selection for the ``anthropic``
                provider.

        Returns:
            A ``dict`` always containing ``timeout`` and ``api_version``, and
            conditionally ``base_url`` / ``api_key`` when resolved.
        """
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
        """
        Build kwargs for the Google Gemini provider.

        Currently identical in shape to the generic/OpenAI builders; kept as
        its own method (rather than relying on ``_build_generic``) so
        Gemini-specific kwargs can be added later without affecting other
        providers.

        Args:
            selection: The resolved provider selection for the ``gemini``
                provider.

        Returns:
            A ``dict`` always containing ``timeout``, and conditionally
            ``base_url`` / ``api_key`` when resolved.
        """
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
        """
        Build kwargs for any provider without a dedicated builder method.

        Used as the fallback in :meth:`build` for providers registered in
        the database that don't yet have bespoke handling here (e.g. a newly
        added provider before its own ``_build_*`` method is written) — so
        adding a new provider to the database does not immediately break
        resolution even before code support for its quirks lands.

        Args:
            selection: The resolved provider selection for an unrecognized
                provider name.

        Returns:
            A ``dict`` always containing ``timeout``, and conditionally
            ``base_url`` / ``api_key`` when resolved.
        """
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
        """
        Resolve the request timeout, preferring the database configuration.

        Args:
            selection: The resolved provider selection, whose
                ``configuration.timeout_seconds`` (if set) takes priority.

        Returns:
            The timeout in seconds as a ``float``: the database-configured
            value if present, otherwise ``Settings.TIMEOUT``.
        """
        if selection.configuration and selection.configuration.timeout_seconds is not None:
            return float(selection.configuration.timeout_seconds)
        return float(self._settings.TIMEOUT)

    def _base_url(
        self,
        selection: ResolvedProviderSelection,
        *,
        fallback: str | None,
    ) -> str | None:
        """
        Resolve the provider base URL, preferring more specific configuration.

        Precedence, most to least specific:

        1. ``ProviderConfiguration.endpoint`` — an explicit admin override
           for this provider (e.g. a self-hosted or proxied endpoint).
        2. ``Provider.base_url`` — the base URL set when the provider was
           registered.
        3. ``fallback`` — a caller-supplied default (e.g.
           ``Settings.OLLAMA_HOST`` for Ollama), or ``None`` for providers
           whose SDK already has a sensible built-in default.

        Args:
            selection: The resolved provider selection to read configuration
                from.
            fallback: Value to use when neither database source has a base
                URL. Pass ``None`` for providers that should rely on their
                SDK's own default when unconfigured.

        Returns:
            The resolved base URL, or ``None`` if no source (including
            ``fallback``) provides one.
        """
        if selection.configuration and selection.configuration.endpoint:
            return selection.configuration.endpoint
        if selection.provider and selection.provider.base_url:
            return selection.provider.base_url
        return fallback

    def _read_api_key(self, selection: ResolvedProviderSelection) -> str | None:
        """
        Read the actual API key secret value from the environment.

        The database only ever stores the *name* of an environment variable
        (``APIKey.api_key_env`` / ``ProviderConfiguration.api_key_env``),
        never the secret itself — this method is where that name is finally
        turned into a real value, at the last possible moment before it is
        handed to a provider adapter.

        Two candidate environment variable names are tried, in order: the
        dedicated ``APIKey`` row's env var first (a specific, admin-managed
        credential), then the ``ProviderConfiguration`` row's env var as a
        secondary source (a more general per-provider default). The first
        environment variable that is actually set (non-empty) wins.

        Args:
            selection: The resolved provider selection, whose ``api_key`` and
                ``configuration`` fields may each name an environment
                variable to check.

        Returns:
            The API key value read from the environment, or ``None`` if
            neither candidate environment variable is set.
        """
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
        """
        Map of provider name to its dedicated kwargs-builder method.

        Used by :meth:`build` for dispatch; providers not present here fall
        back to :meth:`_build_generic`.

        Returns:
            A ``dict`` mapping provider name strings (e.g. ``"ollama"``) to
            bound builder methods.
        """
        return {
            "ollama": self._build_ollama,
            "openai": self._build_openai,
            "anthropic": self._build_anthropic,
            "gemini": self._build_gemini,
        }
