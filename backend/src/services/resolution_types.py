"""
Shared value types for database-backed provider/model resolution.

This module is the vocabulary shared by the resolution pipeline:

- ``ResolutionSource``          — *where* a resolved value came from (used for
                                   logging/observability and precedence checks).
- ``ResolvedApiKey``            — a non-secret reference to an API key row.
- ``ResolvedProviderSelection`` — the intermediate result of
                                   :class:`~src.services.provider_resolver.ProviderResolver`
                                   (provider/model identity plus the raw
                                   database rows needed to build config).
- ``ResolvedConfiguration``     — the final, cache-ready output of
                                   :class:`~src.services.provider_resolution_coordinator.ProviderResolutionCoordinator`,
                                   containing everything
                                   :class:`~src.providers.factory.ProviderFactory.create`
                                   needs.

These are plain, dependency-light dataclasses — they hold no behavior beyond
construction and are passed between the resolver, the config resolver, the
coordinator, and the cache (`src/core/provider_resolution_cache.py`) as
immutable snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.models.api_key import APIKey
from src.models.provider import Provider
from src.models.provider_configuration import ProviderConfiguration


class ResolutionSource(str, Enum):
    """
    Identifies which precedence tier produced a resolved provider or model value.

    The gateway resolves "which provider/model do we use?" by trying sources
    in a fixed precedence order — an explicit request value wins first, then a
    database-configured default, then a settings/environment default, and
    finally a hardcoded fallback. This enum records which tier actually won,
    so callers (and logs) can explain *why* a particular provider was chosen
    without re-deriving the precedence chain.

    Being a ``str`` subclass means members compare equal to their plain string
    value and serialize cleanly (e.g. ``resolved.provider_source.value`` in a
    log line), without any extra conversion step.

    Members:
        REQUEST: The value was supplied explicitly by the caller/HTTP request
            (e.g. ``provider="openai"`` in the chat payload). Always takes
            priority over any stored default.
        DATABASE: The value came from a row in the database (e.g. the
            provider marked ``is_default=True``, or a model's
            ``get_default_model`` result).
        SETTINGS: The value came from application settings/environment
            variables (``Settings.DEFAULT_PROVIDER`` / ``DEFAULT_MODEL``),
            used when no database default exists.
        FALLBACK: The value came from a hardcoded constant in code
            (``HARDCODED_PROVIDER_FALLBACK`` in ``provider_resolver.py``),
            used only when no request, database, or settings value is
            available — a last resort so resolution never fails outright.

    Example:
        >>> ResolutionSource.DATABASE == "database"
        True
        >>> ResolutionSource.DATABASE.value
        'database'
    """

    REQUEST = "request"
    DATABASE = "database"
    SETTINGS = "settings"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ResolvedApiKey:
    """
    Non-secret reference to a resolved API key, safe to log or pass around.

    This deliberately does **not** carry the secret key value itself — only
    the name of the environment variable that holds it (``env_var``) and a
    human-readable label (``key_identifier``). The actual secret is read from
    the environment later, at the point of use
    (see ``ProviderConfigResolver._read_api_key``), so this object can be
    freely logged or cached without leaking credentials.

    Attributes:
        env_var: Name of the environment variable holding the API key (e.g.
            ``"OPENAI_API_KEY"``), or ``None`` if no key row was found for
            the provider.
        key_identifier: A non-secret admin-facing label for the key (e.g.
            ``"prod-primary"``), or ``None`` if unavailable.

    Example:
        >>> ref = ResolvedApiKey(env_var="OPENAI_API_KEY", key_identifier="prod-primary")
        >>> ref.env_var
        'OPENAI_API_KEY'
    """

    env_var: str | None
    key_identifier: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProviderSelection:
    """
    Intermediate result of provider/model identity resolution.

    Produced by :meth:`~src.services.provider_resolver.ProviderResolver.resolve`
    and consumed by :meth:`~src.services.provider_config_resolver.ProviderConfigResolver.build`,
    which turns this identity information into concrete constructor kwargs
    for :class:`~src.providers.factory.ProviderFactory`. Splitting these two
    steps keeps "which provider/model?" (identity + precedence) separate from
    "how do we connect to it?" (transport config).

    Attributes:
        provider_name: The resolved provider's stable machine key (e.g.
            ``"ollama"``, ``"openai"``). Always present, even if no matching
            database row exists yet.
        model_name: The resolved model identifier to send to the provider
            (e.g. ``"qwen3:8b"``, ``"gpt-4o"``).
        provider_source: Which precedence tier produced ``provider_name``
            (see :class:`ResolutionSource`).
        model_source: Which precedence tier produced ``model_name``.
        provider: The matching :class:`~src.models.provider.Provider` row, or
            ``None`` if the resolved provider name has no database entry
            (e.g. a request-supplied provider name that hasn't been
            registered yet).
        configuration: The provider's :class:`~src.models.provider_configuration.ProviderConfiguration`
            row (connection settings), or ``None`` if there is no database
            provider row or no configuration has been saved for it.
        api_key: The provider's default active :class:`~src.models.api_key.APIKey`
            row, or ``None`` if none exists (e.g. local providers like Ollama
            that don't require a key).

    Example:
        >>> selection = ResolvedProviderSelection(
        ...     provider_name="ollama",
        ...     model_name="qwen3:8b",
        ...     provider_source=ResolutionSource.SETTINGS,
        ...     model_source=ResolutionSource.SETTINGS,
        ... )
        >>> selection.provider is None
        True
    """

    provider_name: str
    model_name: str
    provider_source: ResolutionSource
    model_source: ResolutionSource
    provider: Provider | None = None
    configuration: ProviderConfiguration | None = None
    api_key: APIKey | None = None


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    """
    Final, cache-ready resolution result consumed by ``ProviderFactory.create()``.

    This is the output of :class:`~src.services.provider_resolution_coordinator.ProviderResolutionCoordinator.resolve`:
    a fully-formed, database-free snapshot that no longer references live ORM
    rows (unlike :class:`ResolvedProviderSelection`), which makes it safe to
    hold in an in-memory TTL cache
    (:class:`~src.core.provider_resolution_cache.ProviderResolutionCache`)
    without pinning a database session or stale ORM objects in memory.

    Attributes:
        provider_name: The resolved provider's stable machine key (e.g.
            ``"ollama"``).
        model_name: The resolved model identifier (e.g. ``"qwen3:8b"``).
        factory_kwargs: Provider-specific keyword arguments (e.g.
            ``base_url``, ``timeout``, ``api_key``) ready to be unpacked into
            :meth:`~src.providers.factory.ProviderFactory.create`.
        provider_source: Which precedence tier produced ``provider_name``
            (see :class:`ResolutionSource`).
        model_source: Which precedence tier produced ``model_name``.

    Example:
        >>> resolved = ResolvedConfiguration(
        ...     provider_name="ollama",
        ...     model_name="qwen3:8b",
        ...     factory_kwargs={"base_url": "http://localhost:11434", "timeout": 30.0},
        ...     provider_source=ResolutionSource.SETTINGS,
        ...     model_source=ResolutionSource.SETTINGS,
        ... )
        >>> resolved.factory_kwargs["timeout"]
        30.0
    """

    provider_name: str
    model_name: str
    factory_kwargs: dict[str, Any]
    provider_source: ResolutionSource
    model_source: ResolutionSource
