"""
Centralized application configuration for the AI Gateway.

Per the "Configuration" section of the AI Gateway architecture rules, all
provider settings, server settings, database settings, and other tunables
live here (in ``core/config.py``) rather than being scattered across
services or hardcoded at call sites. Values are loaded from environment
variables (and optional ``.env`` files) via ``pydantic-settings``, so
secrets never need to be hardcoded in source.

Other layers should obtain configuration through the module-level
``settings`` singleton (or the ``get_settings()`` accessor for testability),
never by constructing ``Settings()`` directly, so the whole process shares
one consistent, cached configuration snapshot.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# `parents[2]` walks up from this file (backend/src/core/config.py) to the
# `backend/` directory: config.py -> core -> src -> backend.
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """
    Typed, environment-driven configuration for the entire backend process.

    Each field maps to an environment variable of the same name (e.g.
    ``DATABASE_URL`` in the environment populates ``Settings.DATABASE_URL``).
    Values are validated and coerced to their declared types by Pydantic at
    startup, so a malformed environment variable (e.g. a non-numeric
    ``PORT``) fails fast with a clear error instead of causing confusing
    runtime behavior later.

    Attributes:
        APP_NAME: Human-readable service name shown in the OpenAPI docs and
            the root ``/`` endpoint response. Defaults to ``"AI Gateway"``.
        APP_VERSION: Semantic version string surfaced in the OpenAPI docs and
            root endpoint, e.g. ``"0.1.0"``.
        HOST: Network interface the server binds to. ``"0.0.0.0"`` (all
            interfaces) is the default so the service is reachable inside
            containers without extra configuration.
        PORT: TCP port the server listens on. Defaults to ``8000``.
        OLLAMA_HOST: Base URL of the local/self-hosted Ollama server used by
            ``OllamaService``, e.g. ``"http://localhost:11434"``.
        OLLAMA_MODEL: Default Ollama model tag to use when no model is
            otherwise specified, e.g. ``"qwen3:8b"``.
        DEFAULT_PROVIDER: Provider name used when a chat request does not
            specify one, e.g. ``"ollama"``. Consumed by provider resolution
            (see ``services/provider_resolution_coordinator.py``).
        DEFAULT_MODEL: Model name used when a chat request does not specify
            one, paired with ``DEFAULT_PROVIDER``.
        PROVIDER_RESOLUTION_CACHE_TTL_SECONDS: How long (in seconds) a
            resolved provider/model configuration stays valid in
            ``ProviderResolutionCache`` before it must be re-resolved from
            the database/settings. A short TTL (default ``60.0``) balances
            reducing repeated DB lookups against picking up configuration
            changes reasonably quickly.
        PROVIDER_HTTP_MAX_RETRIES: Maximum number of retry attempts for
            transient HTTP failures when calling a provider's API. Defaults
            to ``3`` to tolerate brief network blips without retrying
            indefinitely.
        PROVIDER_HTTP_RETRY_BASE_DELAY_SECONDS: Base delay (seconds) used by
            the provider HTTP client's backoff strategy between retries,
            e.g. for exponential backoff starting at ``0.5`` seconds.
        PROVIDER_HTTP_RETRY_MAX_DELAY_SECONDS: Upper bound (seconds) on the
            backoff delay between retries, preventing exponential backoff
            from growing unbounded (capped at ``8.0`` seconds by default).
        PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS: How long (seconds) a
            provider's list of available models is cached before being
            re-fetched, e.g. ``300.0`` (5 minutes). Model catalogs change
            infrequently, so a longer TTL than the resolution cache is
            appropriate here.
        ANTHROPIC_API_VERSION: The ``anthropic-version`` header value sent
            with every Anthropic API request, e.g. ``"2023-06-01"``. Pinned
            explicitly so provider behavior does not silently change when
            Anthropic ships a new API version.
        LOG_LEVEL: Root log level name (e.g. ``"INFO"``, ``"DEBUG"``) used by
            ``core/logging.py`` when configuring logging at startup.
        TIMEOUT: Timeout, in seconds, applied to outbound provider HTTP
            requests (e.g. Ollama). Defaults to ``60`` seconds to allow for
            slower local model inference while still failing eventually.
        DATABASE_URL: SQLAlchemy async connection string for the primary
            database, e.g. ``"postgresql+asyncpg://user:pass@host/db"``. Has
            no default — startup fails if it is not supplied, since the
            gateway cannot function without a database connection.
        TEST_DATABASE_URL: Optional separate connection string used by the
            test suite so tests never run against the primary database.
            ``None`` means tests fall back to ``DATABASE_URL`` (see test
            fixtures for the exact fallback logic).
        DB_POOL_SIZE: Number of persistent connections SQLAlchemy's async
            engine keeps open in its connection pool. Defaults to ``5``.
        DB_MAX_OVERFLOW: Number of additional, temporary connections the
            pool may open beyond ``DB_POOL_SIZE`` under load before new
            requests must wait. Defaults to ``10``.
        DB_ECHO: When ``True``, SQLAlchemy logs every SQL statement it
            executes. Useful for local debugging only — the security rules
            require this stay ``False`` (the default) in production to
            avoid leaking query details/parameters into logs.
        model_config: Pydantic-settings configuration controlling how values
            are loaded — see the inline comments below for details on the
            ``.env`` file search order and ``extra="ignore"``.

    Example:
        >>> from src.core.config import get_settings
        >>> s = get_settings()
        >>> isinstance(s.PORT, int)
        True
    """

    # Application
    APP_NAME: str = "AI Gateway"
    APP_VERSION: str = "0.1.0"
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"

    # Provider defaults and resolution cache
    DEFAULT_PROVIDER: str = "ollama"
    DEFAULT_MODEL: str = "qwen3:8b"
    PROVIDER_RESOLUTION_CACHE_TTL_SECONDS: float = 60.0

    # Provider HTTP retry and model cache
    PROVIDER_HTTP_MAX_RETRIES: int = 3
    PROVIDER_HTTP_RETRY_BASE_DELAY_SECONDS: float = 0.5
    PROVIDER_HTTP_RETRY_MAX_DELAY_SECONDS: float = 8.0
    PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS: float = 300.0
    ANTHROPIC_API_VERSION: str = "2023-06-01"

    # Logging
    LOG_LEVEL: str = "INFO"

    TIMEOUT: int = 60  # Timeout for requests in seconds

    # Database
    DATABASE_URL: str
    TEST_DATABASE_URL: str | None = None
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    model_config = SettingsConfigDict(
        # Root settings are shared by Docker and the application. The backend file
        # remains available for backend-specific overrides such as Ollama settings.
        # Both files are listed so pydantic-settings loads the root `.env` first
        # and lets a backend-local `.env` override/supplement it if present.
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        # Ignore unknown environment variables (e.g. ones meant for Docker or
        # other services sharing the root `.env`) instead of raising a
        # validation error for every extra key.
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Build (once) and return the process-wide ``Settings`` instance.

    Wrapped in ``functools.lru_cache`` so ``Settings()`` — which reads
    environment variables and ``.env`` files — is only constructed once per
    process, no matter how many times ``get_settings()`` is called. This
    keeps configuration lookups cheap and guarantees every caller sees the
    exact same values for the lifetime of the process.

    Returns:
        The single, cached ``Settings`` instance for this process.

    Raises:
        pydantic.ValidationError: If a required field (e.g. ``DATABASE_URL``)
            is missing from the environment/``.env`` files, or if any field
            fails type coercion (e.g. a non-numeric ``PORT``).

    Example:
        >>> from src.core.config import get_settings
        >>> get_settings() is get_settings()  # same cached instance
        True
    """
    return Settings()

# Module-level singleton so most call sites can simply `from src.core.config
# import settings` instead of calling `get_settings()` everywhere. Tests that
# need to override configuration should still prefer `get_settings()` (e.g.
# with dependency overrides) since `settings` is bound at import time.
settings = get_settings()
