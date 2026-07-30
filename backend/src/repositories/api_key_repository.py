"""
Persistence queries for :class:`~src.models.api_key.APIKey`.

An ``APIKey`` row is a *reference* to a provider credential — it stores the
name of the environment variable holding the secret (``api_key_env``), never
the secret value itself. Beyond generic CRUD from
:class:`~src.repositories.base.BaseRepository`, this module adds
provider-scoped lookups: by logical name, by "the" default key for a
provider, and a listing of active keys.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_key import APIKey
from src.repositories.base import BaseRepository


class APIKeyRepository(BaseRepository[APIKey]):
    """
    Data access for provider credential references.

    Extends :class:`~src.repositories.base.BaseRepository` with
    provider-scoped queries that rely on the database constraints defined
    in ``models/api_key.py``: a composite unique key
    (``uq_api_keys_provider_id_name``) and a partial unique index limiting
    each provider to at most one default key
    (``uq_api_keys_one_default_per_provider``).
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to the ``APIKey`` model.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = APIKeyRepository(session)
        """
        super().__init__(session, APIKey)

    async def get_by_provider_and_name(
        self,
        provider_id: int,
        name: str,
    ) -> APIKey | None:
        """
        Fetch a credential row by provider and logical name.

        Filters on the composite ``(provider_id, name)`` pair because
        ``name`` (e.g. ``"prod-primary"``) is only unique *within* a
        provider (see ``uq_api_keys_provider_id_name`` in
        ``models/api_key.py``) — different providers may reuse the same
        logical key name.

        Args:
            provider_id: Primary key of the owning
                :class:`~src.models.provider.Provider`.
            name: The credential's logical name/label, e.g.
                ``"prod-primary"``. Not a secret — used for admin UI
                display and lookup only.

        Returns:
            The matching ``APIKey``, or ``None`` if no credential with that
            name is registered under that provider.

        Example:
            >>> key = await repo.get_by_provider_and_name(1, "prod-primary")
            >>> key is None or key.name == "prod-primary"
            True
        """
        return await self.get_one(
            APIKey.provider_id == provider_id,
            APIKey.name == name,
        )

    async def get_default_for_provider(self, provider_id: int) -> APIKey | None:
        """
        Return the default active key for a provider, if configured.

        The partial unique index ``uq_api_keys_one_default_per_provider``
        guarantees at most one row per provider can have
        ``is_default=True``, so this query can never return an ambiguous
        result even without an explicit ``LIMIT``.

        Args:
            provider_id: Primary key of the
                :class:`~src.models.provider.Provider` to look up the
                default credential for.

        Returns:
            The active default ``APIKey`` for that provider, or ``None`` if
            no key is marked default (or the default key has since been
            deactivated).

        Example:
            >>> key = await repo.get_default_for_provider(1)
            >>> key is None or key.is_default
            True
        """
        return await self.get_one(
            APIKey.provider_id == provider_id,
            APIKey.is_default.is_(True),
            APIKey.is_active.is_(True),
        )

    async def list_active_for_provider(
        self,
        provider_id: int,
        *,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[APIKey]:
        """
        Return active keys for a provider ordered by name.

        Args:
            provider_id: Primary key of the owning
                :class:`~src.models.provider.Provider`.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``APIKey`` rows for that provider where ``is_active``
            is ``True``, sorted alphabetically by ``name``. Empty list if
            the provider has no active keys.

        Example:
            >>> keys = await repo.list_active_for_provider(1)
            >>> all(k.provider_id == 1 and k.is_active for k in keys)
            True
        """
        return await self.list(
            APIKey.provider_id == provider_id,
            APIKey.is_active.is_(True),
            order_by=(APIKey.name.asc(),),
            offset=offset,
            limit=limit,
        )
