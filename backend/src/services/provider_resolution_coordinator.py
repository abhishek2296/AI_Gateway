"""
Coordinate provider resolution, config building, and caching for AIService.

This module is the single entry point :class:`~src.services.ai_service.AIService`
uses to go from "what provider/model did the caller ask for?" to "here are
the ready-to-use factory kwargs" (:class:`~src.services.resolution_types.ResolvedConfiguration`).
It wires together three collaborators on every cache miss:

1. :class:`~src.services.provider_resolver.ProviderResolver` — determines
   provider/model identity via the request > database > settings > fallback
   precedence chain.
2. :class:`~src.services.provider_config_resolver.ProviderConfigResolver` —
   turns that identity into provider-specific constructor kwargs.
3. :class:`~src.core.provider_resolution_cache.ProviderResolutionCache` — an
   in-memory TTL cache that avoids repeating the (potentially several)
   database round trips resolution requires on every single chat/health
   request.

``ProviderResolutionCoordinator`` is the layer that owns *when* a database
session is opened — resolution's database repositories are only ever
constructed inside a short-lived ``async with`` block scoped to a single
cache miss, so callers of this coordinator never need to manage sessions
themselves.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.config import Settings, get_settings
from src.core.provider_resolution_cache import ProviderResolutionCache
from src.repositories.ai_model_repository import AIModelRepository
from src.repositories.api_key_repository import APIKeyRepository
from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
from src.repositories.provider_repository import ProviderRepository
from src.services.provider_config_resolver import ProviderConfigResolver
from src.services.provider_resolver import ProviderResolver
from src.services.resolution_types import ResolvedConfiguration


class ProviderResolutionCoordinator:
    """
    Resolve provider configuration with optional TTL caching.

    Opens a short-lived database session per cache miss (via the injected
    ``session_factory``) rather than holding one open for the coordinator's
    lifetime, so a long-lived coordinator instance never pins a stale
    connection or ORM session — this keeps the coordinator, and by extension
    :class:`~src.services.ai_service.AIService`, framework-agnostic and safe
    to reuse across many requests.

    The cache is keyed on the *raw request hints* (``request_provider``,
    ``request_model``) rather than the resolved values, so a cache hit
    completely skips the database and the resolver — only the previously
    computed :class:`~src.services.resolution_types.ResolvedConfiguration` is
    returned. This trades a small amount of staleness (changes to database
    defaults take up to ``PROVIDER_RESOLUTION_CACHE_TTL_SECONDS`` to be
    picked up) for avoiding several database round trips on every request.

    Example:
        >>> coordinator = ProviderResolutionCoordinator(session_factory=AsyncSessionLocal)
        >>> resolved = await coordinator.resolve(request_provider="ollama")
        >>> resolved.provider_name
        'ollama'
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
        cache: ProviderResolutionCache | None = None,
        config_resolver: ProviderConfigResolver | None = None,
    ) -> None:
        """
        Wire up the session factory, settings, cache, and config resolver.

        All dependencies except ``session_factory`` are optional and default
        to sensible shared instances, which keeps simple call sites (e.g.
        ``create_ai_service``) terse while still allowing tests to inject
        fakes for the cache or config resolver.

        Args:
            session_factory: An ``async_sessionmaker`` used to open a new
                database session on each cache miss. Required because there
                is no safe default — the caller must supply the app's
                configured session factory (e.g. ``AsyncSessionLocal``).
            settings: Application settings. Defaults to
                :func:`~src.core.config.get_settings` when omitted, so the
                coordinator picks up the process-wide configuration
                automatically.
            cache: The TTL cache instance to use. Defaults to a new
                :class:`~src.core.provider_resolution_cache.ProviderResolutionCache`
                sized by ``Settings.PROVIDER_RESOLUTION_CACHE_TTL_SECONDS``.
                Overridable so tests can inject a cache with a known/zero TTL.
            config_resolver: The config resolver used to build factory
                kwargs. Defaults to a new
                :class:`~src.services.provider_config_resolver.ProviderConfigResolver`.
                Overridable for tests that want to stub kwargs building.
        """
        self._settings = settings or get_settings()
        self._session_factory = session_factory
        self._cache = cache or ProviderResolutionCache(
            self._settings.PROVIDER_RESOLUTION_CACHE_TTL_SECONDS,
        )
        self._config_resolver = config_resolver or ProviderConfigResolver(self._settings)

    async def resolve(
        self,
        *,
        request_provider: str | None = None,
        request_model: str | None = None,
    ) -> ResolvedConfiguration:
        """
        Resolve provider configuration, using the cache when possible.

        On a cache hit, this method never touches the database — it returns
        the previously computed result immediately. On a cache miss, it opens
        a database session, runs the full
        :class:`~src.services.provider_resolver.ProviderResolver` resolution
        chain, builds factory kwargs via
        :class:`~src.services.provider_config_resolver.ProviderConfigResolver`,
        and stores the result in the cache before returning it — so the next
        call with the same ``(request_provider, request_model)`` pair, within
        the TTL window, is served from memory.

        Args:
            request_provider: Provider name explicitly supplied by the
                caller, or ``None`` to use the database/settings/fallback
                precedence tiers. Also used verbatim as (half of) the cache
                key, so two logically-equivalent-but-differently-cased
                strings (e.g. ``"OpenAI"`` vs ``"openai"``) would be treated
                as distinct cache entries.
            request_model: Model name explicitly supplied by the caller, or
                ``None`` to use the database/settings precedence tiers.

        Returns:
            A :class:`~src.services.resolution_types.ResolvedConfiguration`
            containing the resolved provider/model names, ready-to-use
            factory kwargs, and which precedence tier each value came from.

        Raises:
            ProviderDisabledError: If ``request_provider`` names a disabled
                database provider (propagated from
                :meth:`~src.services.provider_resolver.ProviderResolver.resolve`).
            ProviderResolutionError: If no provider can be resolved from any
                precedence tier.

        Example:
            >>> resolved = await coordinator.resolve(request_provider=None, request_model=None)
            >>> resolved.provider_source
            <ResolutionSource.SETTINGS: 'settings'>
        """
        cached = self._cache.get(request_provider, request_model)
        if cached is not None:
            return cached

        # Scope the database session tightly to just the resolution work: it
        # is opened only for the (possibly several) repository lookups the
        # resolver needs, and is closed again before this method returns —
        # nothing downstream (the cache, the caller) ever sees a live session.
        async with self._session_factory() as session:
            resolver = ProviderResolver(
                provider_repository=ProviderRepository(session),
                model_repository=AIModelRepository(session),
                api_key_repository=APIKeyRepository(session),
                configuration_repository=ProviderConfigurationRepository(session),
                settings=self._settings,
            )
            selection = await resolver.resolve(
                request_provider=request_provider,
                request_model=request_model,
            )

        factory_kwargs = self._config_resolver.build(selection)
        resolved = ResolvedConfiguration(
            provider_name=selection.provider_name,
            model_name=selection.model_name,
            factory_kwargs=factory_kwargs,
            provider_source=selection.provider_source,
            model_source=selection.model_source,
        )
        self._cache.set(request_provider, request_model, resolved)
        return resolved
