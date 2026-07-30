"""
Persistence queries for :class:`~src.models.prompt_template.PromptTemplate`.

A ``PromptTemplate`` row is a versioned, reusable prompt (name + version +
template body) not tied to any specific chat session. Beyond generic CRUD
from :class:`~src.repositories.base.BaseRepository`, this module adds
lookup by the template's composite natural key (name + version) and a
category-filterable listing of active templates.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.prompt_template import PromptTemplate
from src.repositories.base import BaseRepository


class PromptTemplateRepository(BaseRepository[PromptTemplate]):
    """
    Data access for versioned prompt templates.

    Extends :class:`~src.repositories.base.BaseRepository` with a
    composite-key lookup (``get_by_name_and_version``) and an
    ``is_active`` + optional-category listing (``list_active``).
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Bind this repository to a session, scoped to ``PromptTemplate``.

        Args:
            session: The active ``AsyncSession`` used for all queries issued
                by this repository instance.

        Example:
            >>> repo = PromptTemplateRepository(session)
        """
        super().__init__(session, PromptTemplate)

    async def get_by_name_and_version(
        self,
        name: str,
        version: int,
    ) -> PromptTemplate | None:
        """
        Fetch a template by unique name and version.

        Filters on the composite ``(name, version)`` pair because templates
        are versioned: the same ``name`` can have multiple rows, one per
        ``version``, enforced unique together by
        ``uq_prompt_templates_name_version`` in ``models/prompt_template.py``.
        Use this when a caller needs an exact, pinned version rather than
        "whatever the latest active version is" (see ``list_active`` for
        that case).

        Args:
            name: The template's logical name, e.g.
                ``"summarize_conversation"``.
            version: The specific version number to fetch, e.g. ``2``.

        Returns:
            The matching ``PromptTemplate``, or ``None`` if no template with
            that name/version pair exists.

        Example:
            >>> template = await repo.get_by_name_and_version("summarize_conversation", 2)
            >>> template is None or template.version == 2
            True
        """
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
        """
        Return active templates, optionally filtered by category.

        Ordered by name ascending, then version *descending*, so that when
        multiple versions of the same template are active, the newest
        version for each name appears first within that name's group —
        convenient for callers that want to display or default to "the
        latest version" per template name.

        Args:
            category: If given, restrict results to templates with this
                exact ``category`` value (e.g. ``"summarization"``).
                Keyword-only. ``None`` (the default) returns active
                templates across every category.
            offset: Number of matching rows to skip, for pagination.
                Keyword-only. ``None`` means no offset.
            limit: Maximum number of rows to return. Keyword-only. ``None``
                means no limit.

        Returns:
            A list of ``PromptTemplate`` rows where ``is_active`` is
            ``True`` (and, if given, ``category`` matches), ordered by name
            ascending then version descending. Empty list if none match.

        Example:
            >>> templates = await repo.list_active(category="summarization")
            >>> all(t.is_active for t in templates)
            True
        """
        filters = [PromptTemplate.is_active.is_(True)]
        if category is not None:
            filters.append(PromptTemplate.category == category)
        return await self.list(
            *filters,
            order_by=(PromptTemplate.name.asc(), PromptTemplate.version.desc()),
            offset=offset,
            limit=limit,
        )
