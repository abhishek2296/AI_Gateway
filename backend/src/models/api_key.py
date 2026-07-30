"""
ORM entity for provider API credentials (env-var references only).

This module deliberately never stores an actual secret value. Per
``08-security.mdc``, all secrets must live in environment variables — this
table only records *which* environment variable holds a given provider's
key, plus non-secret bookkeeping metadata (a label, expiry, last-used time).
Allowing multiple rows per provider supports key rotation and multi-tenant
setups (e.g. separate keys per environment or customer) without schema
changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint, false, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.provider import Provider


class APIKey(Base, TimestampMixin):
    """
    Credential reference for a :class:`~src.models.provider.Provider`.

    Maps to the ``api_keys`` table. Represents metadata about a credential
    used to authenticate with a provider — never the credential itself.
    ``api_key_env`` stores the **environment variable name** (e.g.
    ``OPENAI_API_KEY``), never the secret value. ``key_identifier`` is a
    non-secret label for admin UI (e.g. ``prod-primary``, ``last4:…8f2a``),
    distinct from ``name`` which is the row's own identifying label.
    At most one row per provider may set ``is_default=True``, enforced by a
    partial unique index so the invariant holds regardless of which code
    path writes to the table.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        provider_id: Foreign key to ``providers.id``. ``ondelete="CASCADE"``
            — a credential reference is meaningless without its provider,
            so it is removed automatically when the provider is deleted.
            Indexed.
        name: Human-assigned label for this key row, e.g. ``"primary"`` or
            ``"backup-2026"``. ``String(255)``, required. Combined with
            ``provider_id`` in a unique constraint so a provider cannot
            have two keys with the same name.
        key_identifier: Non-secret display identifier for admin UIs (e.g.
            a masked suffix like ``"last4:8f2a"``), safe to show without
            revealing the credential. ``String(255)``, indexed.
        api_key_env: Name of the environment variable holding the actual
            secret value (e.g. ``"OPENAI_API_KEY"``). ``String(255)``,
            required. The application resolves this at call time via
            ``os.environ`` (or ``Settings``); the secret itself is never
            persisted to the database.
        organization: Optional provider-side organization id/slug this key
            is scoped to (relevant for providers like OpenAI that support
            multiple orgs per account). ``String(255)``, nullable.
        project: Optional provider-side project id/slug this key is scoped
            to. ``String(255)``, nullable.
        is_default: Whether this is the provider's default key, used when
            a request doesn't specify one explicitly. ``Boolean``, defaults
            to ``False``, indexed. At most one ``True`` row per
            ``provider_id`` — enforced by the partial unique index
            ``uq_api_keys_one_default_per_provider`` (see
            ``__table_args__``).
        is_active: Whether this key is currently usable. ``Boolean``,
            defaults to ``True``, indexed. Set to ``False`` to retire a key
            (e.g. after rotation) without deleting its row/history.
        expires_at: Optional expiry timestamp after which this key should
            no longer be used. ``DateTime(timezone=True)``, nullable —
            ``None`` means "no known expiry".
        last_used_at: Timestamp this key was last used successfully,
            useful for auditing and detecting stale/unused credentials.
            ``DateTime(timezone=True)``, nullable — ``None`` until first
            use. Updated by application code, not a database trigger.
        provider: Many-to-one back to :class:`~src.models.provider.Provider`.
            Eagerly loaded (``lazy="selectin"``).

    Example:
        >>> key = APIKey(
        ...     provider_id=1,
        ...     name="primary",
        ...     key_identifier="last4:8f2a",
        ...     api_key_env="OPENAI_API_KEY",
        ...     is_default=True,
        ... )
        >>> key.api_key_env
        'OPENAI_API_KEY'
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("provider_id", "name", name="uq_api_keys_provider_id_name"),
        # Partial unique index: only rows where is_default is True are
        # constrained to be unique per provider_id, so any number of
        # non-default keys can coexist for the same provider.
        Index(
            "uq_api_keys_one_default_per_provider",
            "provider_id",
            unique=True,
            postgresql_where=text("is_default IS TRUE"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255))
    key_identifier: Mapped[str] = mapped_column(String(255), index=True)

    api_key_env: Mapped[str] = mapped_column(String(255))
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True)
    project: Mapped[str | None] = mapped_column(String(255), nullable=True)

    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="api_keys",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<APIKey id=1 name='primary' provider_id=1>``.
            Only the non-secret ``name`` label is included — neither
            ``api_key_env`` nor ``key_identifier`` are shown, keeping this
            safe to log even though neither field is itself a secret.

        Example:
            >>> APIKey(id=1, name="primary", provider_id=1).__repr__()
            "<APIKey id=1 name='primary' provider_id=1>"
        """
        return (
            f"<APIKey id={self.id} name={self.name!r} provider_id={self.provider_id}>"
        )
