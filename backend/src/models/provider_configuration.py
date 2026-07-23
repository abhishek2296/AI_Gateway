"""ORM entity for provider-specific connection and runtime settings."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.provider import Provider


class ProviderConfiguration(Base, TimestampMixin):
    """
    One-to-one connection settings for a :class:`~src.models.provider.Provider`.

    ``api_key_env`` stores the **environment variable name** that holds the
    secret (e.g. ``OPENAI_API_KEY``), never the secret value itself.
    Provider-specific knobs that do not warrant dedicated columns go in
    ``extra_config``.
    """

    __tablename__ = "provider_configurations"
    __table_args__ = (
        UniqueConstraint("provider_id", name="uq_provider_configurations_provider_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    api_key_env: Mapped[str | None] = mapped_column(String(255), nullable=True)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True)
    region: Mapped[str | None] = mapped_column(String(100), nullable=True)
    project: Mapped[str | None] = mapped_column(String(255), nullable=True)

    timeout_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_retries: Mapped[int | None] = mapped_column(Integer, nullable=True)

    verify_ssl: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
    )
    proxy_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    extra_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="configuration",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<ProviderConfiguration id={self.id} provider_id={self.provider_id}>"
        )
