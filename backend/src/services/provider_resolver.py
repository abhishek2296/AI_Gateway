"""
Database-backed provider and model resolution for :class:`~src.services.ai_service.AIService`.

This module answers the question "which provider and which model should
handle this request?" It sits between the route/service layer and the
database, and implements the gateway's precedence rules for choosing a
provider and model:

1. ``request``  — an explicit value supplied by the caller (e.g. the HTTP
   request body) always wins, because the caller knows best what they want.
2. ``database`` — otherwise, fall back to whatever is configured as the
   default in the database (the provider marked ``is_default=True``, or a
   model flagged as an ``AIModel`` default for that provider).
3. ``settings`` — otherwise, fall back to the application's static
   configuration (``Settings.DEFAULT_PROVIDER`` / ``DEFAULT_MODEL``).
4. ``fallback`` — as an absolute last resort, use a hardcoded constant
   (``HARDCODED_PROVIDER_FALLBACK``) so the gateway can still function even
   with an empty database and no settings configured.

:class:`ProviderResolver` is intentionally framework-agnostic (it only
depends on repositories and settings, not FastAPI) so it can be reused from
CLI tools, background workers, or any future entry point — not just HTTP
routes. The result of resolution (:class:`~src.services.resolution_types.ResolvedProviderSelection`)
is then handed to :class:`~src.services.provider_config_resolver.ProviderConfigResolver`
to build provider-specific connection kwargs.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from src.core.config import Settings
from src.models.provider import Provider
from src.repositories.ai_model_repository import AIModelRepository
from src.repositories.api_key_repository import APIKeyRepository
from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
from src.repositories.provider_repository import ProviderRepository
from src.services.resolution_types import (
    ResolutionSource,
    ResolvedApiKey,
    ResolvedProviderSelection,
)

logger = logging.getLogger(__name__)

# Absolute last-resort provider name used only when the database has no
# default provider row *and* Settings.DEFAULT_PROVIDER is unset/empty — keeps
# the gateway usable on a fresh install with zero configuration.
HARDCODED_PROVIDER_FALLBACK = "ollama"

# A "provider name strategy" is a zero-argument async callable that attempts
# to produce a provider name from one precedence tier (database, settings,
# fallback), returning None if that tier has nothing to offer. Modeling each
# tier as a callable lets `_resolve_provider_name` iterate a plain tuple
# instead of a chain of if/elif branches.
ProviderNameStrategy = Callable[[], Awaitable[str | None]]


class ProviderResolutionError(Exception):
    """
    Raised when provider or model resolution fails before any adapter is invoked.

    This signals a configuration/data problem (e.g. no provider could be
    resolved from any precedence tier) rather than a runtime failure talking
    to a provider backend. Callers such as :class:`~src.services.ai_service.AIService`
    catch this and translate it into a normalized application exception
    (``LLMResponseException``) before it reaches the HTTP layer.

    Example:
        try:
            selection = await resolver.resolve()
        except ProviderResolutionError:
            # No provider/model could be determined at all.
            raise LLMResponseException()
    """


class ProviderDisabledError(ProviderResolutionError):
    """
    Raised when a request explicitly names a provider that exists but is disabled.

    This is distinct from "no provider found" (:class:`ProviderResolutionError`):
    here the provider *is* registered in the database, but its
    ``Provider.is_active`` flag is ``False``. It is only raised for an
    explicit ``request_provider`` — when resolution falls back to a database
    default, a disabled provider is silently skipped instead (see
    ``_resolve_provider_name``), since a stale default should not hard-fail
    the request.

    Example:
        # Provider "openai" exists in the DB but is_active=False.
        await resolver.resolve(request_provider="openai")
        # -> raises ProviderDisabledError("Provider 'openai' is disabled.")
    """


class ProviderResolver:
    """
    Resolve provider identity, model name, and credential references for one request.

    A single ``ProviderResolver`` instance is scoped to one database session
    (via the injected repositories) and one resolution call — it is cheap to
    construct and is typically created fresh per request by
    :class:`~src.services.provider_resolution_coordinator.ProviderResolutionCoordinator`.

    Resolution follows the ordered precedence chain described in the module
    docstring (request > database > settings > fallback), implemented as a
    tuple of ``(ResolutionSource, strategy)`` pairs in
    ``self._provider_name_strategies`` rather than nested ``if``/``elif``
    branches — this makes the precedence order visible at a glance and easy
    to extend (e.g. inserting a new tier) without restructuring conditionals.

    Being framework-agnostic (no FastAPI imports, only repositories and
    settings), it is safe to reuse from CLI tools, background workers, or any
    future entry point beyond HTTP routes.

    Example:
        >>> resolver = ProviderResolver(
        ...     provider_repository=ProviderRepository(session),
        ...     model_repository=AIModelRepository(session),
        ...     api_key_repository=APIKeyRepository(session),
        ...     configuration_repository=ProviderConfigurationRepository(session),
        ...     settings=get_settings(),
        ... )
        >>> selection = await resolver.resolve(request_provider="openai")
    """

    def __init__(
        self,
        provider_repository: ProviderRepository,
        model_repository: AIModelRepository,
        api_key_repository: APIKeyRepository,
        configuration_repository: ProviderConfigurationRepository,
        settings: Settings,
    ) -> None:
        """
        Wire up the repositories and settings this resolver reads from.

        Args:
            provider_repository: Data access for ``Provider`` rows (lookup by
                name, default provider lookup).
            model_repository: Data access for ``AIModel`` rows (default model
                lookup per provider).
            api_key_repository: Data access for ``APIKey`` rows (default
                active key lookup per provider).
            configuration_repository: Data access for
                ``ProviderConfiguration`` rows (connection settings per
                provider).
            settings: Application settings, used for the ``settings`` and
                default-model precedence tiers (``DEFAULT_PROVIDER``,
                ``DEFAULT_MODEL``).
        """
        self._provider_repository = provider_repository
        self._model_repository = model_repository
        self._api_key_repository = api_key_repository
        self._configuration_repository = configuration_repository
        self._settings = settings
        # Ordered precedence chain for resolving a provider *name* when the
        # caller did not supply one explicitly. Each entry pairs the source
        # label (for logging/telemetry) with the strategy that produces it;
        # `_resolve_provider_name` walks this tuple in order and stops at the
        # first strategy that returns a usable, non-disabled provider.
        self._provider_name_strategies: tuple[tuple[ResolutionSource, ProviderNameStrategy], ...] = (
            (ResolutionSource.DATABASE, self._resolve_database_default_provider),
            (ResolutionSource.SETTINGS, self._resolve_settings_provider),
            (ResolutionSource.FALLBACK, self._resolve_hardcoded_provider),
        )

    async def resolve(
        self,
        *,
        request_provider: str | None = None,
        request_model: str | None = None,
    ) -> ResolvedProviderSelection:
        """
        Resolve the provider and model to use, plus their supporting database rows.

        This is the main entry point for the resolver: it determines the
        provider name and model name (each independently following the
        request > database > settings > fallback precedence chain), then
        loads the provider's connection configuration and default API key so
        downstream code (:class:`~src.services.provider_config_resolver.ProviderConfigResolver`)
        has everything it needs to build factory kwargs in one round trip.

        Args:
            request_provider: Provider name explicitly supplied by the
                caller (e.g. from the HTTP request body), or ``None``/empty
                to fall through to the database/settings/fallback tiers.
            request_model: Model name explicitly supplied by the caller, or
                ``None``/empty to fall through to the database/settings
                tiers. Resolved independently of ``request_provider`` — a
                caller may specify only one of the two.

        Returns:
            A :class:`~src.services.resolution_types.ResolvedProviderSelection`
            describing the chosen provider/model, which precedence tier each
            came from, and (when available) the matching ``Provider``,
            ``ProviderConfiguration``, and ``APIKey`` rows.

        Raises:
            ProviderDisabledError: If ``request_provider`` names a provider
                that exists in the database but has ``is_active=False``.
            ProviderResolutionError: If no provider name can be determined
                from any precedence tier (should only happen if
                ``Settings.DEFAULT_PROVIDER`` is empty and the database has
                no default provider, since ``HARDCODED_PROVIDER_FALLBACK``
                normally guarantees a result).

        Example:
            >>> selection = await resolver.resolve(request_provider="openai", request_model="gpt-4o")
            >>> selection.provider_source
            <ResolutionSource.REQUEST: 'request'>
        """
        provider_name, provider_source, provider_row = await self._resolve_provider_name(
            request_provider,
        )
        model_name, model_source = await self._resolve_model_name(
            request_model=request_model,
            provider_row=provider_row,
            provider_name=provider_name,
        )
        configuration = await self._load_configuration(provider_row)
        api_key = await self._load_api_key(provider_row)

        logger.info(
            "Resolved provider=%s (source=%s), model=%s (source=%s)",
            provider_name,
            provider_source.value,
            model_name,
            model_source.value,
        )
        return ResolvedProviderSelection(
            provider_name=provider_name,
            model_name=model_name,
            provider_source=provider_source,
            model_source=model_source,
            provider=provider_row,
            configuration=configuration,
            api_key=api_key,
        )

    async def resolve_api_key(self, provider: Provider | None) -> ResolvedApiKey | None:
        """
        Return a non-secret reference to a provider's default active API key.

        This is a convenience method for callers that already have a
        ``Provider`` row (e.g. from an earlier resolution) and just need the
        credential reference, without going through the full ``resolve()``
        flow again. It deliberately returns a
        :class:`~src.services.resolution_types.ResolvedApiKey` — never the
        raw secret value — so the result stays safe to log or cache.

        Args:
            provider: The provider to look up a key for, or ``None`` if no
                provider row is available (e.g. an unregistered
                request-supplied provider name).

        Returns:
            A :class:`~src.services.resolution_types.ResolvedApiKey` wrapping
            the key's environment variable name and identifier, or ``None``
            if ``provider`` is ``None`` or has no default active key.

        Example:
            >>> ref = await resolver.resolve_api_key(provider_row)
            >>> ref.env_var if ref else None
            'OPENAI_API_KEY'
        """
        row = await self._load_api_key(provider)
        if row is None:
            return None
        return ResolvedApiKey(env_var=row.api_key_env, key_identifier=row.key_identifier)

    async def _load_api_key(self, provider: Provider | None):
        """
        Fetch the default active ``APIKey`` row for a provider, if any.

        Args:
            provider: The provider to look up, or ``None`` when no provider
                row exists (in which case there is nothing to look up).

        Returns:
            The ``APIKey`` ORM row, or ``None`` if ``provider`` is ``None``
            or the provider has no default active key configured.
        """
        if provider is None:
            return None
        return await self._api_key_repository.get_default_for_provider(provider.id)

    async def _resolve_provider_name(
        self,
        request_provider: str | None,
    ) -> tuple[str, ResolutionSource, Provider | None]:
        """
        Determine the provider name, its resolution source, and its database row.

        This implements the provider half of the request > database >
        settings > fallback precedence chain:

        - If ``request_provider`` is given, it always wins. If it matches a
          database row that is disabled, resolution fails loudly
          (:class:`ProviderDisabledError`) rather than silently falling back,
          because an explicit request for a disabled provider is very likely
          a caller/config mistake that should surface immediately.
        - Otherwise, ``self._provider_name_strategies`` is walked in order
          (database default -> settings default -> hardcoded fallback).
          Within this fallback path, a disabled database row is *skipped*
          rather than raised — a stale "default provider" flag should
          degrade gracefully to the next tier instead of breaking every
          request that didn't ask for a specific provider.

        Args:
            request_provider: Provider name explicitly supplied by the
                caller, or ``None``/empty to use the precedence chain.

        Returns:
            A 3-tuple of ``(provider_name, source, provider_row)`` where
            ``provider_row`` is the matching ``Provider`` ORM row, or
            ``None`` if the resolved name has no database entry.

        Raises:
            ProviderDisabledError: If ``request_provider`` matches a database
                row with ``is_active=False``.
            ProviderResolutionError: If ``request_provider`` is empty and no
                strategy in the precedence chain yields a name.
        """
        if request_provider:
            row = await self._provider_repository.get_by_name(request_provider)
            if row is not None and not row.is_active:
                logger.warning("Requested provider %r is disabled", request_provider)
                raise ProviderDisabledError(
                    f"Provider {request_provider!r} is disabled.",
                )
            if row is not None:
                logger.info("Using request provider %r from database", row.name)
                return row.name, ResolutionSource.REQUEST, row
            logger.info("Using request provider %r without database row", request_provider)
            return request_provider, ResolutionSource.REQUEST, None

        for source, strategy in self._provider_name_strategies:
            name = await strategy()
            if not name:
                continue
            row = await self._provider_repository.get_by_name(name)
            if row is not None and not row.is_active:
                # A disabled provider should not block resolution when it was
                # only reached via a fallback tier (not requested explicitly)
                # -- move on to the next, lower-priority strategy instead.
                logger.warning("Skipping disabled provider %r during resolution", name)
                continue
            if row is not None:
                # Deliberately reported as DATABASE (not `source`, which may be
                # SETTINGS or FALLBACK): once a matching, active Provider row
                # exists, the *database* is what confirmed/enriched the name,
                # so downstream consumers get an accurate "this provider is
                # registered" signal alongside the actual precedence tier
                # that supplied the name in the first place (visible via the
                # original `source` in logs above).
                return name, ResolutionSource.DATABASE, row
            return name, source, None

        raise ProviderResolutionError("Unable to resolve a provider.")

    async def _resolve_model_name(
        self,
        *,
        request_model: str | None,
        provider_row: Provider | None,
        provider_name: str,
    ) -> tuple[str, ResolutionSource]:
        """
        Determine the model name and its resolution source.

        Implements the model half of the precedence chain, which is simpler
        than the provider chain since there is no "disabled model" concept
        and no hardcoded model fallback tier:

        1. ``request_model``, if given, always wins.
        2. Otherwise, if a database ``provider_row`` is available, look up
           that provider's default model. Note this is intentionally scoped
           to the *already-resolved* provider — a default model only makes
           sense in the context of a specific provider, so this step is
           skipped entirely when no provider row exists (e.g. an
           unregistered request-supplied provider name).
        3. Otherwise, fall back to ``Settings.DEFAULT_MODEL``, which is
           assumed to always be configured (unlike the provider chain, there
           is no further hardcoded fallback here).

        Args:
            request_model: Model name explicitly supplied by the caller, or
                ``None``/empty to use the precedence chain.
            provider_row: The already-resolved ``Provider`` database row (see
                ``_resolve_provider_name``), or ``None`` if the provider has
                no database entry. Used to scope the database default-model
                lookup.
            provider_name: The already-resolved provider name, used only for
                log messages (to say *which* provider's default model was
                used).

        Returns:
            A 2-tuple of ``(model_name, source)``.

        Example:
            >>> await resolver._resolve_model_name(
            ...     request_model=None, provider_row=ollama_row, provider_name="ollama",
            ... )
            ('qwen3:8b', <ResolutionSource.DATABASE: 'database'>)
        """
        if request_model:
            logger.info("Using request model %r", request_model)
            return request_model, ResolutionSource.REQUEST

        if provider_row is not None:
            default_model = await self._model_repository.get_default_model(provider_row.id)
            if default_model is not None:
                logger.info(
                    "Using database default model %r for provider %r",
                    default_model.model_name,
                    provider_name,
                )
                return default_model.model_name, ResolutionSource.DATABASE

        logger.info("Using settings default model %r", self._settings.DEFAULT_MODEL)
        return self._settings.DEFAULT_MODEL, ResolutionSource.SETTINGS

    async def _load_configuration(self, provider: Provider | None):
        """
        Fetch the ``ProviderConfiguration`` row for a provider, if any.

        Args:
            provider: The provider to look up, or ``None`` when no provider
                row exists.

        Returns:
            The ``ProviderConfiguration`` ORM row, or ``None`` if ``provider``
            is ``None`` or the provider has no saved configuration.
        """
        if provider is None:
            return None
        return await self._configuration_repository.get_by_provider_id(provider.id)

    async def _resolve_database_default_provider(self) -> str | None:
        """
        Strategy: look up the provider row flagged as the database default.

        This is the first (highest-priority) strategy tried in
        ``self._provider_name_strategies`` when no request provider was
        given.

        Returns:
            The default provider's name, or ``None`` if no provider is
            currently marked as the default in the database.
        """
        row = await self._provider_repository.get_default_provider()
        if row is None:
            return None
        logger.info("Using database default provider %r", row.name)
        return row.name

    async def _resolve_settings_provider(self) -> str | None:
        """
        Strategy: use the statically configured default provider from settings.

        Tried after the database-default strategy fails (or is skipped due to
        being disabled). Unlike the database strategy, this never returns
        ``None`` unless ``Settings.DEFAULT_PROVIDER`` itself is falsy.

        Returns:
            ``Settings.DEFAULT_PROVIDER``.
        """
        logger.info("Using settings default provider %r", self._settings.DEFAULT_PROVIDER)
        return self._settings.DEFAULT_PROVIDER

    async def _resolve_hardcoded_provider(self) -> str | None:
        """
        Strategy: use the hardcoded fallback provider name.

        This is the last strategy in the chain, guaranteeing that
        ``_resolve_provider_name`` always has *some* provider name to try
        even on a completely unconfigured install (empty database, empty
        settings) — it is what keeps ``ProviderResolutionError`` from being
        the common case.

        Returns:
            ``HARDCODED_PROVIDER_FALLBACK`` (``"ollama"``).
        """
        logger.info("Using hardcoded provider fallback %r", HARDCODED_PROVIDER_FALLBACK)
        return HARDCODED_PROVIDER_FALLBACK
