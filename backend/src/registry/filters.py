"""
Shared list filters for model registry queries.

Every registry backend (memory, Redis, database) can reuse
:class:`ModelListFilters` so filter semantics stay identical regardless of
where catalog data is stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.registry.models import ModelCapability, ModelInfo, ProviderType


@dataclass(frozen=True, slots=True)
class ModelListFilters:
    """
    Optional constraints applied when listing catalog entries.

    Unset fields (``None``) mean "do not filter on this attribute". When
    multiple fields are set, a model must satisfy **all** of them (AND logic).

    Attributes:
        provider: Return only models belonging to this backend family.
        capability: Return only models that advertise this capability.
        enabled: When ``True`` or ``False``, restrict to enabled/disabled models.
            When ``None``, both enabled and disabled models are included.
        streaming: Convenience flag for ``ModelCapability.STREAMING``. When
            ``True``, only streaming-capable models match. When ``False``,
            only models **without** streaming capability match. When ``None``,
            streaming is not considered.

    Example:
        >>> filters = ModelListFilters(provider=ProviderType.OLLAMA, enabled=True)
        >>> filters.matches(some_ollama_model)
        True
    """

    provider: ProviderType | None = None
    capability: ModelCapability | None = None
    enabled: bool | None = None
    streaming: bool | None = None

    def matches(self, model: ModelInfo) -> bool:
        """
        Return whether ``model`` satisfies every active filter in this object.

        Args:
            model: One catalog entry to test.

        Returns:
            ``True`` when the model passes all non-``None`` filters.

        Example:
            >>> f = ModelListFilters(streaming=True)
            >>> f.matches(model_with_streaming_capability)
            True
        """
        if self.provider is not None and model.provider != self.provider:
            return False

        if self.capability is not None and not model.supports(self.capability):
            return False

        if self.enabled is not None and model.enabled != self.enabled:
            return False

        if self.streaming is not None:
            has_streaming = model.supports(ModelCapability.STREAMING)
            if self.streaming and not has_streaming:
                return False
            if not self.streaming and has_streaming:
                return False

        return True

    @classmethod
    def from_kwargs(
        cls,
        *,
        provider: ProviderType | None = None,
        capability: ModelCapability | None = None,
        enabled: bool | None = None,
        streaming: bool | None = None,
    ) -> ModelListFilters | None:
        """
        Build filters from keyword arguments, or return ``None`` if all are unset.

        Registry ``list()`` methods accept optional filter kwargs. This helper
        converts them into a :class:`ModelListFilters` instance so callers can
        write ``filters = ModelListFilters.from_kwargs(**kwargs)`` once.

        Returns:
            A filter object when at least one argument is not ``None``; otherwise
            ``None`` meaning "no filtering".
        """
        if (
            provider is None
            and capability is None
            and enabled is None
            and streaming is None
        ):
            return None
        return cls(
            provider=provider,
            capability=capability,
            enabled=enabled,
            streaming=streaming,
        )
