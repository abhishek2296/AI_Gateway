"""
Abstract model-registry contract for the gateway catalog.

Phase 6.1 introduced :class:`~src.registry.models.ModelInfo` — the *shape* of
one catalog entry. This module defines *how* those entries are stored and
looked up without committing to a specific backend.

Why an abstract class?
----------------------
The gateway will eventually need more than one way to hold the catalog:

- **In-memory** — fast, good for tests and single-process dev.
- **Redis** — shared cache across multiple API instances (Phase 10+).
- **Database** — durable source of truth synced with the ``AIModel`` ORM table.

If routes and services talked directly to a concrete ``dict`` or SQLAlchemy
session, switching backends would force wide refactors. An abstract
:class:`BaseModelRegistry` gives every layer one stable interface
(``register``, ``get``, ``list``, …) while each backend supplies its own
storage logic in a subclass.

Nothing in this file performs I/O. Concrete implementations live in future
phases (e.g. ``MemoryModelRegistry``, ``RedisModelRegistry``,
``DatabaseModelRegistry``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.registry.models import ModelCapability, ModelInfo, ProviderType


class BaseModelRegistry(ABC):
    """
    Abstract interface for storing and querying :class:`ModelInfo` catalog entries.

    Every model in the registry is uniquely identified by the pair
    ``(provider, name)`` — for example ``(ProviderType.OPENAI, "gpt-4o")``.
    Two different providers may reuse the same model name string without
    conflict because the provider disambiguates them.

    Subclasses must implement all abstract methods. Callers (routing services,
    admin APIs, sync jobs) depend on this contract, not on how data is stored.

    Example:
        A future in-memory implementation might look like this (simplified)::

            class MemoryModelRegistry(BaseModelRegistry):
                def __init__(self) -> None:
                    self._models: dict[tuple[ProviderType, str], ModelInfo] = {}

                async def register(self, model: ModelInfo) -> None:
                    key = (model.provider, model.name)
                    if key in self._models:
                        raise ModelAlreadyRegisteredError(...)
                    self._models[key] = model

                async def get(self, provider, name) -> ModelInfo:
                    ...

        Services then accept ``BaseModelRegistry`` in their constructor and work
        with any backend without code changes.
    """

    @abstractmethod
    async def register(self, model: ModelInfo) -> None:
        """
        Add one model to the catalog.

        Why this method exists:
            New models arrive from provider ``list_models()`` sync jobs, admin
            tooling, or seed scripts. ``register`` is the single entry point for
            putting validated :class:`ModelInfo` records into the registry.

        Args:
            model: A fully validated catalog entry. The registry keys it by
                ``(model.provider, model.name)``.

        Raises:
            ModelAlreadyRegisteredError: If a model with the same provider and
                name is already present (implementations should not silently
                overwrite unless a future subclass documents different semantics).
            InvalidModelMetadataError: If ``model`` fails validation (raised by
                :class:`ModelInfo` construction, not usually by the registry itself).

        Example:
            >>> # Pseudocode — requires a concrete subclass:
            >>> await registry.register(ModelInfo(
            ...     name="gpt-4o",
            ...     provider=ProviderType.OPENAI,
            ...     display_name="GPT-4o",
            ...     description="Flagship multimodal model.",
            ...     capabilities=frozenset({ModelCapability.CHAT}),
            ... ))
        """

    @abstractmethod
    async def unregister(self, provider: ProviderType, name: str) -> None:
        """
        Remove one model from the catalog.

        Why this method exists:
            Models are retired, disabled at the provider, or removed during
            catalog cleanup. ``unregister`` deletes the entry so ``get`` and
            ``list`` no longer return it.

        Args:
            provider: Backend family the model belongs to.
            name: Provider-facing model identifier (same string stored on
                :attr:`ModelInfo.name`).

        Raises:
            ModelNotFoundError: If no model exists for the given provider/name.

        Example:
            >>> await registry.unregister(ProviderType.OPENAI, "gpt-4o")
        """

    @abstractmethod
    async def get(self, provider: ProviderType, name: str) -> ModelInfo:
        """
        Fetch one model by provider and name.

        Why this method exists:
            Routing and resolution need a single catalog record — for example
            to read ``context_window``, ``capabilities``, or ``enabled`` before
            forwarding a request to a provider adapter.

        Args:
            provider: Backend family to look under.
            name: Model identifier within that provider.

        Returns:
            The matching :class:`ModelInfo` instance.

        Raises:
            ModelNotFoundError: If the model is not in the registry.

        Example:
            >>> model = await registry.get(ProviderType.OPENAI, "gpt-4o")
            >>> model.display_name
            'GPT-4o'
        """

    @abstractmethod
    async def list(
        self,
        *,
        provider: ProviderType | None = None,
        capability: ModelCapability | None = None,
        enabled: bool | None = None,
        streaming: bool | None = None,
    ) -> Sequence[ModelInfo]:
        """
        Return catalog entries, optionally filtered.

        Why this method exists:
            Admin UIs, ``GET /models`` endpoints, and routing engines need the
            full catalog or a filtered slice (by provider, capability, enabled
            flag, etc.) without each caller re-implementing filter logic.

        Args:
            provider: When set, only models for this backend family are returned.
            capability: When set, only models advertising this capability match.
            enabled: When set, restrict to enabled (``True``) or disabled
                (``False``) models.
            streaming: When ``True``, only models with ``STREAMING`` capability;
                when ``False``, only models without it; when ``None``, ignore.

        Returns:
            A sequence of matching :class:`ModelInfo` records. Order is
            implementation-defined; callers should sort if they need a stable
            display order. Unspecified filter kwargs return the full catalog.

        Example:
            >>> models = await registry.list(provider=ProviderType.OLLAMA, enabled=True)
            >>> models = await registry.list(capability=ModelCapability.CHAT, streaming=True)
        """

    @abstractmethod
    async def exists(self, provider: ProviderType, name: str) -> bool:
        """
        Check whether a model is registered without fetching it.

        Why this method exists:
            Sync jobs and validators often only need a yes/no answer ("should I
            insert or update?") without loading the full :class:`ModelInfo`.
            Cheaper than :meth:`get` when the caller handles absence differently
            than presence.

        Args:
            provider: Backend family to check.
            name: Model identifier within that provider.

        Returns:
            ``True`` if the model is registered, ``False`` otherwise.

        Example:
            >>> if await registry.exists(ProviderType.OLLAMA, "qwen3:8b"):
            ...     print("Already cataloged")
        """

    @abstractmethod
    async def providers(self) -> Sequence[ProviderType]:
        """
        List provider families that have at least one registered model.

        Why this method exists:
            UIs and health dashboards group models by vendor. This method
            answers "which backends appear in our catalog?" without scanning
            the full :meth:`list` output and deduplicating manually.

        Returns:
            A sequence of distinct :class:`ProviderType` values that currently
            have one or more models in the registry. Order is
            implementation-defined.

        Example:
            >>> providers = await registry.providers()
            >>> ProviderType.OPENAI in providers
            True
        """

    @abstractmethod
    async def clear(self) -> None:
        """
        Remove every model from the catalog.

        Why this method exists:
            Tests need a clean slate between cases. Bulk reload jobs may wipe
            and repopulate the catalog from an external source. Production
            code should use this carefully — it is intentionally destructive.

        Example:
            >>> await registry.clear()
            >>> await registry.list()
            ()
        """
