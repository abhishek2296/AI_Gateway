"""Persistence queries for :class:`~src.models.prompt_template.PromptTemplate`."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.prompt_template import PromptTemplate
from src.repositories.base import BaseRepository


class PromptTemplateRepository(BaseRepository[PromptTemplate]):
    """Data access for versioned prompt templates."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, PromptTemplate)

    async def get_by_name_and_version(
        self,
        name: str,
        version: int,
    ) -> PromptTemplate | None:
        """Fetch a template by unique name and version."""
        return await self.get_one(
            PromptTemplate.name == name,
            PromptTemplate.version == version,
        )

    async def list_active(
        self,
        *,
        category: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[PromptTemplate]:
        """Return active templates, optionally filtered by category."""
        filters = [PromptTemplate.is_active.is_(True)]
        if category is not None:
            filters.append(PromptTemplate.category == category)
        return await self.list(
            *filters,
            order_by=(PromptTemplate.name.asc(), PromptTemplate.version.desc()),
            offset=offset,
            limit=limit,
        )
