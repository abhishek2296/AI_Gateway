"""Unit tests for registry list filters."""

from __future__ import annotations

from src.registry.filters import ModelListFilters
from src.registry.models import ModelCapability, ModelInfo, ProviderType


def _model(**kwargs: object) -> ModelInfo:
    defaults = {
        "name": "test",
        "provider": ProviderType.OLLAMA,
        "display_name": "Test",
        "description": "",
        "capabilities": frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
        "enabled": True,
    }
    defaults.update(kwargs)
    return ModelInfo(**defaults)  # type: ignore[arg-type]


def test_filters_and_combination() -> None:
    model = _model()
    assert ModelListFilters(provider=ProviderType.OLLAMA, enabled=True).matches(model)
    assert not ModelListFilters(provider=ProviderType.OPENAI).matches(model)
    assert ModelListFilters(streaming=True).matches(model)
    assert not ModelListFilters(streaming=False).matches(model)
    assert ModelListFilters(capability=ModelCapability.CHAT, streaming=True).matches(model)


def test_from_kwargs_returns_none_when_empty() -> None:
    assert ModelListFilters.from_kwargs() is None
